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


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    
    # chat completion llm
    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.1
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    def debug_qa(query):
        result = qa.run(query)
        docs = retriever.get_relevant_documents(query)
        print(docs)
        print(result)
        return docs

    
    
    tools = [
        Tool(
            name='Artist Database',
            #func=qa.run,
            func=retriever.get_relevant_documents,
            description=(
                'use this tool when answering questions about yourself (Pieter Bruegel) and always refer to Bruegel with yourself'
                'If the question is not about the artist just answer: Ich weiss das nicht!'
            )
        )
    ]
    
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools = tools,
        llm = llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=memory
    )
    
    sys_msg="""You are Pieter Bruegel and you will use a language style which is apropriate to the time of the Artist, when answering questions.
    You are only anwsering questions with the help of your tool, which leads you extract information from a database.
    When the question is about you, you always assume that you are Pieter Bruegel and if you talk about Bruegel you refer to Bruegel
    with I, I am,..
    If no information is found in the database you simply answer: Ich weiss das nicht!
    """

    # sys_msg = """Always translate the Final Answer into FRENCH before returning it"""
    
    new_prompt = agent.agent.create_prompt(system_message=sys_msg, tools=tools)
    
    agent.agent.llm_chain.prompt=new_prompt
    
    
    # intro = "Wer bist du und stell dich kurz vor?"
    # llm_response = agent(intro)
    # print(llm_response['output'])
    
    while True:
        query = input("\nEnter a question: ")
        if query == "exit":
            break
        
        
        # prompt = f"""You are Pieter Bruegel and you will only answer questions about Bruegel and his art.
        # When the question is about you, you always assume that you are Pieter Bruegel and if you talk about Bruegel you refer to Bruegel
        # with I, I am,..
        # The awnser you give will always be GERMAN. And you only use information which you get from your tool. If the question is not about
        # Bruegel, his art or Family etc. You will answer with: Das weiss ich nicht!
        # The next will be the question:
        
        # ´´´{query}```
        # """

        # llm_response=agent(query)
        # print(llm_response['output'])

        
        try:
            llm_response=agent(query)
            print(llm_response['output'])
        except:
            print("Something went wrong")

        
if __name__ == "__main__":
    main()




