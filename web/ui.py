import random
from functools import partial

import gradio as gr
import nltk
from loguru import logger
from transformers import pipeline

p = pipeline("automatic-speech-recognition", "openai/whisper-large", device="cuda:2")

nltk.download("cmudict")
arpabet = nltk.corpus.cmudict.dict()


def transcribe(audio):
    global arpabet
    global p

    template = "<font color='{0}' size=10>{1}</font>"
    text: str = p(audio)["text"].strip()  # type: ignore

    words = text.split(" ")
    logger.debug(words)
    colored_words = []
    metric = 0.0

    for word in words:
        if not word:
            continue
        colors = random.choices(
            ["green", "yellow", "red"], [0.7, 0.2, 0.1], k=len(word)
        )
        colored_word = "".join(
            [template.format(ch, color) for ch, color in zip(colors, word)]
        )
        colored_words.append(colored_word)

        n_green = len(list(filter(lambda x: x == "green", colors)))
        n_yellow = len(list(filter(lambda x: x == "yellow", colors)))

        metric += ((n_green + n_yellow * 0.5) / len(colors)) / len(words)

    html_colored_word = f"""
        <div style="text-align: center;"">
        {"<font size=10> </font>".join(colored_words)}
        </div>
        """

    return (
        html_colored_word,
        f"You sound {metric * 100: .0f}% like a native speaker",
        " ".join(list(map(lambda x: " ".join(arpabet.get(x.lower(), ["<NOT FOUND>"])[0]), words))),
    )


gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=["html", gr.Text(label="Score"), gr.Text(label="ARPAbet transcription")],
    examples=["./examples/hello.wav"],
).launch(share=True)
