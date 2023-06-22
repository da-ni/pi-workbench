from LLM_Pieter import LLM_Pieter
from elevenlabs import generate, play
from elevenlabs import stream

def main():
    pieter_chat_bot = LLM_Pieter()
    
    while True:
        user_input = input("Your message: ")

        if user_input == "exit":
            break

        response = pieter_chat_bot.run_query(user_input)
        print(response)

        """
        audio = generate(
            text= response,
            voice="Bella",
            model="eleven_monolingual_v1"
        )
        
        play(audio)
        """


        audio_stream = generate(
            text=response,
            stream=True,
            model="eleven_multilingual_v1",
            #voice= "jSM24cfFHOkXdXgFH4c9"
            #voice = "pNInz6obpgDQGcFmaJgB"
            voice="VR6AewLTigWG4xSOukaG"
        )

        stream(audio_stream)


if __name__ == "__main__":
    main()
