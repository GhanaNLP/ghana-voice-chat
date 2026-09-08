---
title: Ghana Twi Voice Chat
emoji: 🗣️
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-sa-4.0
short_description: Speak Twi, and a Ghanaian voice answers you in Twi
---

# Ghana Twi Voice Chat

Speak Twi into your microphone. A Ghanaian voice answers you in Twi.

This Space is just the page. The work happens in a [Modal](https://modal.com) backend, on CPU:

| Stage | What runs |
|---|---|
| Speech recognition | [griot-nano-1](https://huggingface.co/Qlerqly/griot-nano-1), a 153M Conformer-CTC for Akan, Dagbani, Ewe, Ga and Ghanaian English, with [KenLM](https://huggingface.co/Qlerqly/griot-nano-1-kenlm) beam search |
| Understanding + reply | Gemini 2.5 Flash, reading the transcript and hearing the recording |
| Voice | [stable-twi-tts](https://github.com/GhanaNLP/stable-twi-tts), ONNX, voice `twi-6` |

The reply is capped at two sentences and each is synthesised and pushed over a websocket as soon
as it is ready, so she starts talking while the rest is still being made.

## Why recognition is a separate model

Gemini cannot transcribe Twi. Given real Twi speech it returned the English sentence *"Why won't
you come out?"* and answered that instead. griot-nano-1 is trained on Ghanaian speech, including
GhanaNLP Community data, so it produces actual Twi.

It is still the hard part of this pipeline. Akan is the model's weakest language — **41.67% word
error rate** with greedy decoding — and its card notes frequent `ɛ/e/a/i` and `ɔ/o/u` confusions
and word-boundary errors. KenLM helps materially: on test audio it turned `ne yaduo no nsia` into
`ne aduonu nsia` (the correct Twi for 26) and split `anoaduro` into `ano aduro`.

So the transcript is given to Gemini as evidence rather than as a quotation, and the prompt tells
it to read through the spelling to the intended meaning. The page shows only her reply; append
`?debug=1` to the URL to see the raw transcript and how the model read it, which is the only way
to tell whether a bad turn was the recogniser's fault or the model's.

## Pointing it somewhere else

The backend URL is the `DEFAULT_API` constant at the top of the script in `index.html`. To try a
different deployment without editing the Space:

```
?api=https://your-workspace--ghana-twi-voice-twivoice-web.modal.run
```

## Licence

The page is yours to reuse. Note that griot-nano-1 and its KenLM model are **CC BY-NC-SA 4.0** —
non-commercial and share-alike — which is why this Space carries that licence too.
