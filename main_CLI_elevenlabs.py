import os
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from LLM_Pieter import LLM_Pieter
from elevenlabs import generate, save
from fastapi.staticfiles import StaticFiles

app = FastAPI()
pieter_chat_bot = None

origins = [
    "127.0.0.1",
    "0.0.0.0",
    "localhost:8000",
    "localhost:4200",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:4200",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str


app.mount("/audio", StaticFiles(directory="./audio"), name="audio")


@app.on_event("startup")
def initialize_pieter_chat_bot():
    try:
        global pieter_chat_bot
        pieter_chat_bot = LLM_Pieter()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def save_audio(response, audio_path):
    audio = generate(
        text=response,
        model="eleven_multilingual_v1",
        voice="VR6AewLTigWG4xSOukaG"
    )
    save(audio, audio_path)
    print(f"> Saved audio to {audio_path}")

@app.post("/chat")
def chat(message: Message, background_tasks: BackgroundTasks):
    try:
        user_input = message.message
        response = pieter_chat_bot.run_query(user_input) # type: ignore

        audio_filename = f"{uuid.uuid4()}.wav"
        audio_dir = "audio"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, audio_filename)

        background_tasks.add_task(save_audio, response, audio_path)

        return {"response": response, "audio_path": audio_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/reset")
def reset_chat_bot():
    try:
        initialize_pieter_chat_bot()
        return {"message": "Chat bot has been reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main_CLI_elevenlabs:app", host="127.0.0.1", port=8000)
