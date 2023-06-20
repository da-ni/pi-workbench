import streamlit as st
from streamlit_chat import message

from LLM_Pieter import LLM_Pieter



def main():

    #streamlit app
    st.set_page_config(page_title="Pieter Bruegel the elder", page_icon="🎨", layout="centered")
    st.header("This is a chatbot about Pieter Bruegel the elder")
    st.title("PieterGPT")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Was willst du wissen"}]

    if "pieter_chat_bot" not in st.session_state:
        st.session_state["pieter_chat_bot"] = LLM_Pieter()

    with st.form("chat_input", clear_on_submit=True):
        a, b = st.columns([4, 1])
        user_input = a.text_input(
            label="Your message:",
            placeholder="What would you like to say?",
            label_visibility="collapsed",
        )
    b.form_submit_button("Send", use_container_width=True)


    for msg in st.session_state.messages:
        message(msg["content"], is_user=msg["role"] == "user")


    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        message(user_input, is_user=True)

        #response = "Bananana" #seq_chain.run({"user_query": user_input})
        response = st.session_state.pieter_chat_bot.run_query({"user_query": user_input})
        #msg = response.choices[0].message
        #st.session_state.messages.append(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)


if __name__ == "__main__":
    main()

