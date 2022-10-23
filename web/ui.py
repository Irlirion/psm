import gradio as gr
from transformers import pipeline

PIPELINE = pipeline("automatic-speech-recognition")


def transcribe(audio):
    global PIPELINE

    text = PIPELINE(audio)["text"]
    return text


gr.Interface(
    fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text"
).launch()
