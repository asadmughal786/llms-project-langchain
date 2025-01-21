from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def speech_to_text(text:str):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    return response.stream_to_file("output.mp3")