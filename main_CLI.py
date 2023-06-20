from LLM_Pieter import LLM_Pieter

def main():
    pieter_chat_bot = LLM_Pieter()
    
    while True:
        user_input = input("Your message: ")

        if user_input == "exit":
            break

        response = pieter_chat_bot.run_query(user_input)
        print(response)


if __name__ == "__main__":
    main()
