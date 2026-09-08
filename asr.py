"""Griot Nano 1 speech recognition, with KenLM beam search.

Wraps Qlerqly/griot-nano-1 -- a 153M Conformer-CTC trained on Akan, Dagbani, Ewe, Ghanaian
English and Ga, partly on GhanaNLP Community data. It replaced asking Gemini to understand Twi
audio directly, which does not work: given real Twi speech Gemini transcribed it as the English
sentence "Why won't you come out?" and then answered that instead.

This is not a solved problem even so. Akan is the model's *worst* language -- 41.67% word error
rate with greedy decoding, and its own card notes frequent e/a/i and o/u confusions plus
word-boundary errors. So the transcript reaching Gemini is a strong hint, not a quotation, and
the prompt downstream is written to treat it that way.

Two decisions worth knowing about:

*KenLM runs from the binary trie, without a unigram list.* pyctcdecode warns that it cannot
recover unigrams from a binary LM and that accuracy "might be reduced", which sounds like it
settles the question. Measured on Twi speech it does not: passing the 524k unigrams extracted
from the ARPA produced identical output to the ARPA itself, and both were *worse* than the plain
binary on one of two clips, inventing non-words ("nknanasi", "nkansakonam") where the binary
read "nipa te nka". The binary also loads by mmap in well under a second against 15-25 s to
digest the unigrams, so it wins on both counts.

*It runs on CPU.* A 24-layer Conformer sounds like it wants a GPU, but 4x subsampling leaves
only ~125 frames for a five-second clip, and inference measures at 0.09-0.19x realtime on six
threads. A GPU would be idle most of the time and cost more than the rest of the app together.
"""
from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
# The model card's suggested starting point, and what its published metrics were measured near.
BEAM_WIDTH = 25
LM_ALPHA = 0.5
LM_BETA = 1.0

# pyctcdecode needs a printable stand-in for the non-emitting tokens, stripped after decoding.
_NON_EMITTING = {"<unk>": "", "<pad>": ""}


class Griot:
    def __init__(self, model_dir: str | Path, lm_path: str | Path | None = None,
                 threads: int | None = None):
        import torch
        from safetensors.torch import load_file

        d = Path(model_dir)
        # The checkpoint ships its own model code; it is not a transformers architecture.
        sys.path.insert(0, str(d / "src"))
        from conformer_ctc.model import ConformerCTC, ConformerCTCConfig

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)

        self.config = ConformerCTCConfig(**json.loads((d / "config.json").read_text()))
        vocab = {str(k): int(v)
                 for k, v in json.loads((d / "vocab.json").read_text()).items()}
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

        model = ConformerCTC(self.config)
        model.load_state_dict(load_file(d / "model.safetensors"))
        self.model = model.to(device="cpu", dtype=torch.float32).eval()

        self.decoder = None
        if lm_path and Path(lm_path).is_file():
            from pyctcdecode import build_ctcdecoder

            labels = [""] * len(vocab)
            for token, index in vocab.items():
                labels[index] = ("" if token == "<blank>"
                                 else _NON_EMITTING.get(token, token))
            self.decoder = build_ctcdecoder(
                labels=labels, kenlm_model_path=str(lm_path),
                alpha=LM_ALPHA, beta=LM_BETA)

    @staticmethod
    def prepare(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Mono float32 at 16 kHz, which is the only thing the feature extractor accepts."""
        from scipy.signal import resample_poly

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        if sample_rate != SAMPLE_RATE:
            g = gcd(sample_rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, sample_rate // g)
        return audio.astype(np.float32, copy=False)

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        from conformer_ctc.data import FeatureConfig, audio_to_log_mel
        from conformer_ctc.model import greedy_decode

        torch = self.torch
        wav = self.prepare(audio, sample_rate)
        if len(wav) < SAMPLE_RATE // 10:
            return ""

        features = audio_to_log_mel(
            wav, SAMPLE_RATE,
            FeatureConfig(sample_rate=SAMPLE_RATE, n_mels=self.config.n_mels))
        lengths = torch.tensor([features.shape[0]], dtype=torch.long)
        with torch.inference_mode():
            out = self.model(features.unsqueeze(0), lengths)

        if self.decoder is not None:
            logits = out.logits[0].float().numpy()
            length = int(out.output_lengths[0].item())
            text = str(self.decoder.decode(logits[:length], beam_width=BEAM_WIDTH))
            for placeholder in _NON_EMITTING.values():
                text = text.replace(placeholder, "")
            return " ".join(text.split())

        return greedy_decode(out.logits, out.output_lengths, self.id_to_token,
                             blank_id=self.config.blank_id,
                             pad_id=self.config.pad_id)[0].strip()
