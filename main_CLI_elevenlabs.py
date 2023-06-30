import os
import uuid
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from LLM_Pieter import LLM_Pieter
from elevenlabs import generate, save

app = FastAPI()
pieter_chat_bot = LLM_Pieter()

class Message(BaseModel):
    message: str

@app.post("/chat")
def chat(message: Message):
    user_input = message.message

    if user_input == "exit":
        return {"response": "Goodbye!"}

    response = pieter_chat_bot.run_query(user_input)

    audio_filename = f"{uuid.uuid4()}.wav"
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, audio_filename)

    audio = generate(
        text=response,
        model="eleven_multilingual_v1",
        voice="VR6AewLTigWG4xSOukaG"
    )

    save(audio, audio_path)

    return {"response": response, "audio_path": audio_path}

if __name__ == "__main__":
    uvicorn.run("main_CLI_elevenlabs:app", host="127.0.0.1", port=8000)