from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import TransformChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import SimpleMemory
from langchain.callbacks import get_openai_callback
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
#from langchain.schema.Document import Document

import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv
import os


#globale variable
vectordb = None



def initialize_vectordb():
    # Load the API keys
    load_dotenv()

    # load the vector store

    #persist_directory = os.environ['PERSIST_DIRECTORY2']
    persist_directory = os.environ['PERSIST_DIRECTORY']
    print(persist_directory)

    embeddings = OpenAIEmbeddings() 

    global vectordb 

    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

#find similar documents used in transform chain
def get_reference_documents(input: dict) -> dict:

    global vectordb

    print("\n\n-----------------------------\nSEARCH STRING: " + input['optimized_query'] + "\n-----------------------------\n\n")

    relevantDocuments = vectordb.similarity_search(input['optimized_query'], k=3)
    #relevantDocuments = vectordb.similarity_search_with_relevance_scores(input['optimized_query']) 
    relevantDocumentsString = ""

    for doc in relevantDocuments:
        relevantDocumentsString += doc.page_content + "\n -------------- \n"

    return {"reference_docs": relevantDocumentsString}







    #response = 
    #           openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    #msg = response.choices[0].message
    #st.session_state.messages.append(msg)
    #message(msg.content)







def main():
    # Load the API keys
    load_dotenv()

    #initialize the chat models
    llmSimilarty = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    llmPieter = ChatOpenAI(temperature=0.5, model='gpt-3.5-turbo')

    #initialize the vector store
    initialize_vectordb()



    #template for the similarty search, now understands multiple languages(italien, spanish, german, english -testet)
    optimize_prompt = PromptTemplate(
        input_variables=["user_query", "history"],
        template ="""The <input_query> needs to be transformed in the following steps:\n\
                    1 Check the language of the input
                    2 Translate the <input> of the user to ENGLISH \n\
                    3 Rephrase the question to the thrid person view, when the user refers to you he means Pieter Bruegel the elder\n\
                    4 Take the previous conversation into account.
                    5 Formulate a <optimized_query> and only output this. \n\n \
                    
                    Here is the previous conversation with the user history: \n
                    {history} \n\n \
                    
                    Here is the example:
                    <input_query>:Wer bist du
                    <optimized_query>:Who is Pieter Bruegel the elder
                    <input_query>:Are you married
                    <optimized_query>:Does Pieter Bruegel the elder have a wife
                    <input_query>:Quién eres
                    <optimized_query>:Who is Pieter Bruegel the elder \n\n \
                    
                    This is the <input_query> you have to work on: \n
                    <input_query>:{user_query} \n\n"""
    )


    #template for the search, with relevat documents
    final_prompt = PromptTemplate(
        #input_variables=["user_query", "reference_docs", "user_query_history", "response_history"],
        input_variables=["user_query", "reference_docs", "history"],
        template ="""You are Pieter Bruegel the elder. You are haveing a conversation with a user 
                    interested in you and your art. \n\
                    Take the following steps to provide the best answer: \n\
                    1 Read the provided Documents in the section >>>DOCUMENTS<<< \n\
                    2 Read the conversation >>>HISTORY<<< with the user for context \n\
                    3 You must generate a detailed response in the linguistic style of people living in the 16th century\
                    in Germany with the information based on the >>>DOCUMENTS<<< and >>>HISTORYY<<<\n\
                    Try not to reapet yourself, but be as eleborate as possible and make sure to use the linguistic style of the 16th century. \n\
                    4 Only answer with the response, without explanation \n\n \
                    
                    This is the conversation >>>HISTORY<<<: \n
                    {history} \n\n \

                    
                    This are the documents >>>DOCUMENTS<<<:\n{reference_docs}\n\n \ 
                    
                    This is the question from the user: \n{user_query} \n\n \

                    <response>:"""
    )


    #set up the single step chains
    optimize_query_chain = LLMChain(llm=llmSimilarty, prompt=optimize_prompt, output_key = 'optimized_query', verbose=False)
    similarity_search_chain = TransformChain(input_variables = ['optimized_query', 'history'], output_variables=['reference_docs'], transform=get_reference_documents)
    final_query_chain = LLMChain(llm=llmPieter, prompt=final_prompt, output_key = 'response', verbose=False)

    #fit the chains together
    seq_chain = SequentialChain(
        memory=ConversationBufferMemory(human_prefix='<input>', ai_prefix='<response>'),
        chains=[optimize_query_chain, similarity_search_chain, final_query_chain], 
        input_variables=['user_query'], 
        output_variables=['response']
        )



    #streamlit app
    st.set_page_config(page_title="Pieter Bruegel the elder", page_icon="🎨", layout="centered")
    st.header("This is a chatbot about Pieter Bruegel the elder")
    st.title("PieterGPT")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Was willst du wissen"}]

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

        response = seq_chain.run({"user_query": user_input})
        #msg = response.choices[0].message
        #st.session_state.messages.append(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)



if __name__ == "__main__":
    main()

