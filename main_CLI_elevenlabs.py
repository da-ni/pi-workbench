import os
import uuid
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from LLM_Pieter import LLM_Pieter
from elevenlabs import generate, save

app = FastAPI()
pieter_chat_bot = None

origins = [
    "127.0.0.1",
    "0.0.0.0",
    "localhost:8000",
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

def initialize_pieter_chat_bot():
    global pieter_chat_bot
    pieter_chat_bot = LLM_Pieter()

@app.post("/chat")
def chat(message: Message):
    user_input = message.message

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

@app.get("/chat/reset")
def reset_chat_bot():
    initialize_pieter_chat_bot()
    return {"message": "Chat bot has been reset."}

if __name__ == "__main__":
    initialize_pieter_chat_bot()
    uvicorn.run("main_CLI_elevenlabs:app", host="127.0.0.1", port=8000)