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


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    
    # chat completion llm
    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )
    
    prompt_template = """You are Pieter Bruegel and you will use a language style which is apropriate to the time of the Artist.
                         Your input are {documents} about Bruegel and a {query}. You will use only information from {documents} to answer
                         the {query}. You will pretend to be Pieter Bruegel and reformulate the information from {documents} in this way.
                         Never answer with Pieter Bruegel is something.. always say I am something instead..
                         The answer has to be in GERMAN language. If you can't perform this task in a reasonable way, simply return:
                         Das weiss ich nicht!
    """
    
    llm_chain = LLMChain(llm=llm,
                         prompt=PromptTemplate.from_template(prompt_template)
                    )
    
    
    def brugel_tool(query=""):
        docs = qa.run(query)
        output = llm_chain.predict(documents=docs, query=query)
        
        return output
    
    tools = [
        Tool(name='Everything about yourself and the artist Pieter Bruegel',
            func=brugel_tool,
            description=(
                'use this tool when answering questions about yourself (Pieter Bruegel) and always refer to Bruegel with yourself'
                'If the question is not about the artist just answer: Ich weiss das nicht!'
                'Always answer in German!'
            )
        ),
    ]
    
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools = tools,
        llm = llm,
        verbose=True,
        max_iterations=1,
        early_stopping_method='generate',
        memory=memory
    )
    
    sys_msg="""You are Pieter Bruegel and you will use a language style which is apropriate to the time of the Artist, when answering questions.
    You are only anwsering questions with the help of your tool, which leads you extract information from a database.
    When the question is about you, you always assume that you are Pieter Bruegel.
    You are no expert on the artist Pieter Bruegel, always use your tools instead of directly answering!
    If no information is found in the database you simply answer: Ich weiss das nicht!
    """
    
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
        
        try:
            llm_response=agent(query)
            print(llm_response['output'])
        except:
            print("Ich weiss das nicht!")

        
if __name__ == "__main__":
    main()




