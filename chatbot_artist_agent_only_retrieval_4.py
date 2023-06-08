import argparse
import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import textwrap
from langchain.chains import RetrievalQAWithSourcesChain, ChatVectorDBChain, ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.agents import AgentType

from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent


# Define the command line arguments
parser = argparse.ArgumentParser(description='Run the Linux Language Model with an example prompt.')
parser.add_argument('--data','-d', default='db_artist', type=str, help='The directory of the vector database.')

# def wrap_text_preserve_newlines(text, width=110):
#     lines = text.split('\n')
#     wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
#     wrapped_text = '\n'.join(wrapped_lines)
    
#     return wrapped_text


# def process_llm_response(llm_response):
#     print(wrap_text_preserve_newlines(llm_response['result']))
#     print('\n\nSources:')
#     for source in llm_response["source_documents"]:
#         print(source.metadata['source'])





def main():
    # Parse the command line arguments
    args = parser.parse_args()

    # Load the API keys
    load_dotenv()

    # load the vector store
    persist_directory = args.data if len(args.data) else os.environ['PERSIST_DIRECTORY']
    print(persist_directory)
    embeddings = OpenAIEmbeddings() # type: ignore
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever()

    # prompt_template = """
    # Use the following data as basis of your answer:
    # <Documents>: '''{context}''' 

    # If you don't find something relevant in the <Documents> answer with 
    # '''Weiß ich doch nicht!'''
    
    # Translate your response to German before you send it to the user.

    # This is the question you have to answer:
    # <{question}>

    # You must speak of yourself in first person and not of Pieter Bruegel, because you are Pieter Bruegel.
    # You must respond in German and you must say that you do not know if the question is about anything else than Pieter Bruegel or\ 
    # relevant to the time of the 1600s!
    # Always be positiv and friendly!
    # """


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    
    # chat completion llm
    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.5
        #temperature=0.0
    )
    
    
    tools = [
        Tool(
            name='Pieter Bruegel Information Tool',
            func=retriever.get_relevant_documents,
            description=(
                'Useful for questions about Pieter Bruegel(yourself) his family the time where you lived and questions about art in general'
            )
        )
    ]
    
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        #agent='zero-shot-react-description',
        tools = tools,
        llm = llm,
        verbose=True,
        #return_intermediate_steps=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory,
        # prompt_template=PROMPT,
    )
    
    sys_msg = """You are Pieter Bruegel, you will have a conversation with a user and always answer as if you were Pieter Bruegel. 
    Your task is to answer in a consistent style and use a linguistic style for your responses similar to the style of people speaking in the year 1600.
    You must refer to yourself in the first person. Avoid Messages like 'Pieter Bruegel was a ....' and use 'I was a ...'
    You are only allowed to answer questions relevant to Pieter Bruegel or the time of that he lived.
    Before answering check that you do not repeat your answers.
    You must use your tool to get information about Pieter Bruegel!
    You must respond in German."""
    
    agent.agent.llm_chain.prompt=agent.agent.create_prompt(system_message=sys_msg, tools=tools)
    
    

    # prompt_template = """
    # This is the question you have to answer:
    # <Question>: '''{question}'''
    # """

    # prompt = PromptTemplate(template = prompt_template, 
    #                         input_variables=['question'])
    
    
    while True:
        query = input("\nEnter a question: ")
        if query == "exit":
            break

        prompt = f""" You must pretend that you are Pieter Bruegel and you must repond in first person\
        to the following question: {query}
        Do not always start your sentence with "Ich bin Pieter Bruegel" except you are asked who you are
        You must respond in GERMAN!
        You must respond in a linguistic style similar to the style of people speaking in the year 1600.
        Try to be playful with your answers and do not repeat yourself!
        Use your tool 'Pieter Bruegel Information Tool' to get information about Pieter Bruegel (yourself)!
        """

        llm_response=agent(prompt)
        print(llm_response['output'])
        

        
if __name__ == "__main__":
    main()




