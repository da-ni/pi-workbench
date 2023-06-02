import argparse
import os
# from dotenv import load_dotenv
import dotenv


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

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text


def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


def main():
    # Parse the command line arguments
    args = parser.parse_args()

    # Load the API keys
    dotenv.load_dotenv()


    # load the vector store
    persist_directory = args.data if len(args.data) else os.environ['PERSIST_DIRECTORY']
    print(persist_directory)
    embeddings = OpenAIEmbeddings() # type: ignore
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )


    # Instantiating the large language model
    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.1
    )

    

    retriever = vectordb.as_retriever()

    prompt_template = """
    You are Pieter Bruegel, you will have a conversation with a user
    Use the following data as basis of your answer:

    >Documents:
    >>>>{context}<<<< 

    Check if the >>>>{question}<<<< is relevat to the provided Documents

    If you don't find something relevant in the >Documents answer with 
    >>>>Weiß ich doch nicht!<<<<

    Use a linguistic style similar to the style of people speaking in the year 1600
    Always speak of Pieter Bruegel in first person perspective     

    
    In General always respond in GERMAN
    """

    PROMPT = PromptTemplate(template = prompt_template, 
                            input_variables=['context', 'question'])
    
    chain_type_kwargs = {"prompt": PROMPT}

    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        # chain_type_kwargs=chain_type_kwargs
        # memory=memory
    )


    qa_no_sources_with_memory = RetrievalQA.from_chain_type(
                                            llm=llm,
                                            chain_type="stuff",
                                            retriever=vectordb.as_retriever(),
                                            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                                        )
    
    qa_no_sources_with_prompt_and_memory = RetrievalQA.from_chain_type(
                                        llm=llm,
                                        chain_type="stuff",
                                        retriever=vectordb.as_retriever(),
                                        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                                        chain_type_kwargs=chain_type_kwargs
                                    )


    while True:
        query = input("\nEnter a question: ")
        if query == "exit":
            break
        
        print("QA with sources:")
        result = qa_with_sources(query)
        print(result['answer'])
        print("Sources:")
        print(result['sources'])

        print("QA without sources, with memory:")
        result = qa_no_sources_with_memory.run(query)
        print(result)

        print("QA without sources, with memory and prompt:")
        result = qa_no_sources_with_prompt_and_memory.run(query)
        print(result)

        
if __name__ == "__main__":
    main()




