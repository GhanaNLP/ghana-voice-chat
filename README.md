# Ghana Twi Voice Chat

Speak Twi into a microphone; a Ghanaian voice answers you in Twi.

- **Live Space** — https://huggingface.co/spaces/ghananlpcommunity/ghana-twi-voice-chat (Docker, CPU)
- **Repo** — https://github.com/GhanaNLP/ghana-voice-chat

## One turn

```
mic (webm/opus)
  -> ffmpeg                 -> 16 kHz mono
  -> griot-nano-1 + KenLM   -> a Twi transcript        Conformer-CTC, on CPU
  -> Gemini 2.5 Flash       -> {understood, reply}     reads a noisy transcript, answers in Twi
  -> stable-twi-tts         -> 22.05 kHz speech        ONNX, voice twi-6
  -> libmp3lame             -> one MP3 per sentence, pushed over the websocket as it is ready
```

Replies are capped at two sentences and each is synthesised as soon as the previous one is sent,
so she starts talking while the rest is still being made. Synthesis runs at about **0.2–0.4x
realtime** and recognition at **0.09–0.19x**, both on CPU. Warm, a turn is about **4 s to first
audio**, most of it the Gemini call.

**Nothing here needs a GPU**, which is the main thing that changed from the Wav2Lip version in
`archive/`. The whole app runs on a CPU Docker Space: cold start is a few seconds (models are
baked into the image).

## Why recognition is a separate model

Gemini cannot transcribe Twi. Given real Twi speech it returned the English sentence *"Why won't
you come out?"* and answered that instead. [griot-nano-1](https://huggingface.co/Qlerqly/griot-nano-1)
is a 153M Conformer-CTC trained on Ghanaian speech including GhanaNLP Community data, so it
produces actual Twi.

It is still the weakest link. Akan is the model's hardest language — **41.67% WER** greedy — and
the transcript reaching Gemini is evidence, not a quotation. The prompt says so explicitly and
tells it to read through the spelling to the intended meaning. That division of labour is what
makes the pipeline work: on one test clip the recogniser produced `me wɔyɛnko` and Gemini still
recovered `Woayɛ ankonam anaa?` — *"are you lonely?"* — because it also hears the recording.

The page shows only her reply. Append `?debug=1` to see the raw transcript and the model's
reading of it — when a turn goes wrong that is the only way to tell which of the two was at
fault, so it is worth reaching for before changing anything.

## Files

| | |
|---|---|
| `app.py` | Standalone FastAPI backend: ASR → Gemini → TTS, WebSocket, serves the frontend |
| `asr.py` | griot-nano-1 wrapper: feature extraction, CTC, KenLM beam search |
| `frontend/` | The page: `index.html` (visualiser, VAD, voice selector) |
| `assets/multilingual.bin` | KenLM binary trie, prebuilt — see below |
| `modal_app.py` | The old Modal backend, kept as reference (no longer deployed) |
| `archive/` | The parked Wav2Lip talking-head pipeline |

## Deploying

This runs as a **Docker Space** on Hugging Face (CPU hardware, so it uses paid credits — not the
free tier). The Space serves both the Flask/FastAPI backend and the static frontend from one
container — no Modal billing.

```bash
# Set GEMINI_API_KEY as a Space secret (Settings → Secrets) — required, the app cannot answer
# without it. Then push the repo:
git clone https://huggingface.co/spaces/ghananlpcommunity/ghana-twi-voice-chat
cp app.py asr.py Dockerfile requirements.txt index.html assets/ .
git add -A && git commit -m "update" && git push
```

Hardware: request `cpu-upgrade` (or a GPU flavor if you want it faster) in the Space settings.
The KenLM binary is baked into the image, so cold start is seconds.

### The old Modal backend

This previously ran on Modal (`modal_app.py`, kept for reference) but Modal bills for idle
containers when the app is kept resident. The Docker Space is self-contained and only runs when
the Space is running. `modal_app.py` remains for anyone who still wants the Modal version; the
Docker/`app.py` version is the maintained path.

## Things that will bite

**`africa-g2p` must stay pinned to 0.1.1.** `ghana-g2p` asks only for `africa-g2p>=0.1.1`, so a
fresh build picks up 0.2.0, which writes affricates with a combining tie bar — `d͡ʑ` (U+0361)
where the voice's symbol table has `dʑ`, and `t͡ɕʰ` where it has `tɕʰ`. Those are the sounds Twi
spells `gy` and `ky`, so with 0.2.0 installed almost every real Twi sentence dies in the
front-end with `PhonemeError` and there is nothing to say.

**Digits are dropped silently by the voice.** `"Me wɔ mfe 25."` is spoken as `me wɔ mfe` — a
different sentence, with no error raised. The prompt forbids digits and `_spell_numbers` catches
the cases where it slips anyway, by asking the model to rewrite its own numerals rather than
hand-rolling Twi number morphology here.

**KenLM ships as a prebuilt binary in `assets/`.** `pip install kenlm` builds the Python
extension but not the `build_binary` executable, so the image cannot convert the published ARPA
itself. The binary is better anyway: 80 MB against 225 MB, and it mmaps instantly where digesting
the ARPA costs 15–25 s of every container's startup. To regenerate, on a machine whose CMake is
older than 4:

```bash
pip install kenlm==0.2.0
hf download Qlerqly/griot-nano-1-kenlm --local-dir lm
build_binary trie lm/multilingual.arpa assets/multilingual.bin
```

**kenlm needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` to compile.** Its CMakeLists declares a pre-3.5
minimum and CMake 4 — which the base image ships — removed compatibility outright. This looks
like a missing dependency and is not; a distro with CMake 3.x builds it without the flag.

**Do not pass KenLM a unigram list.** pyctcdecode warns that it cannot recover unigrams from a
binary LM and that accuracy "might be reduced", which reads like an instruction. Measured on Twi
speech it is wrong: passing the 524k unigrams extracted from the ARPA produced output identical
to the ARPA and *worse* than the plain binary on one of two clips, inventing non-words
(`nknanasi`, `nkansakonam`) where the binary read `nipa te nka`.

**`from __future__ import annotations` breaks FastAPI's websocket injection.** Annotations become
strings, and FastAPI resolves them against *module* globals — so importing `WebSocket` inside the
method that builds the app left `sock: WebSocket` unresolvable, and an unresolvable annotation is
treated as a query parameter. Every connect closed with 1008 `Field required: query.sock`, which
reached the browser as an opaque HTTP 500. The FastAPI imports live in `with image.imports():` at
module scope for this reason.

**Modal's websocket bridge cannot serialise every close frame.** It passes a close message's
`reason` into a protobuf string field without checking it, and Starlette sends a validation
failure's `reason` as a list of dicts — raising `TypeError: bad argument type for built-in
operation` inside Modal's own serialiser, before the handshake completes. `WebsocketCloseShim`
normalises the fields; without it every websocket error is an unreadable 500.

## Endpoints

| | |
|---|---|
| `GET /health` | Voice, ASR model, whether KenLM loaded, beam width, warmup time |
| `GET /say?text=…` | Text → one MP3. No mic, no ASR, no Gemini; isolates the voice |
| `WS /ws` | Send a recording as binary; receive `transcript`, `understood`, `reply`, then `audio` + MP3 pairs |

## Licence

griot-nano-1 and its KenLM model are **CC BY-NC-SA 4.0** — non-commercial and share-alike. That
governs this deployment as a whole, so it is fine as a community demo and not fine inside a paid
product. `stable-twi-tts` is MIT.

## Known rough edges

- Akan recognition is the ceiling on quality. KenLM helps; a larger beam, or fine-tuning on
  cleaner Twi audio, would help more.
- Gemini is asked for Twi, not held to it. It occasionally reaches for an English word, which
  the voice pronounces noticeably worse than Twi.
- The voice selector offers 12 Twi voices (`twi-1` to `twi-12`) ranked by training hours;
  `twi-3` has the most training data (11.2 h), `twi-6` is the current default.
