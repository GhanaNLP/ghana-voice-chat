"""Download the model weights needed to run the app locally.

The app needs three things that are otherwise baked into the Docker image:

  1. griot-nano-1 (the ASR model)  -> ./griot
  2. stable-twi-tts voice          -> cache under the HF hub cache (downloaded automatically
                                      on first use, but do it here so the first start isn't slow)
  3. The KenLM binary trie          -> assets/multilingual.bin

Usage:
    python download_models.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRIOT_DIR = HERE / "griot"


def main() -> None:
    from huggingface_hub import snapshot_download

    print("Downloading griot-nano-1 (ASR) ...")
    snapshot_download("Qlerqly/griot-nano-1", local_dir=str(GRIOT_DIR))

    print("Downloading stable-twi-tts voice ...")
    from stable_twi_tts import StableTwiTTS
    StableTwiTTS.from_pretrained()

    lm = HERE / "assets" / "multilingual.bin"
    if not lm.exists():
        raise SystemExit(
            "\nassets/multilingual.bin (KenLM) is missing.\n"
            "It is a 76 MB prebuilt binary trie. Copy it from a checkout that has it, or build\n"
            "it from the published ARPA (kenlm's build_binary) `<LM_ARPA> assets/multilingual.bin`.\n"
            "Published model: https://huggingface.co/Qlerqly/griot-nano-1-kenlm")

    print("All models present.")


if __name__ == "__main__":
    main()