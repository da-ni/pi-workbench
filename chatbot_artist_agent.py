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
from langchain.chains import RetrievalQAWithSourcesChain, ChatVectorDBChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationBufferMemory


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

    llm = OpenAI(temperature=0.,
                openai_api_key=os.environ['OPENAI_API_KEY'])

    qa = ConversationalRetrievalChain.from_llm(llm,
                                               vectordb.as_retriever(),
                                               memory = memory)
                                            #    return_source_documents=True)
    
    result = qa({'question':'Wer ist Bruegel?'})

    print("hello")
















    docs = vectordb.similarity_search_with_score("Wer ist Bruegel?")

    # memory = ConversationBufferMemory()

    # llm = ChatOpenAI(
    #     openai_api_key=os.environ['OPENAI_API_KEY'],
    #     model_name='gpt-3.5-turbo',
    #     temperature=0.
    #     )


    


    template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Respond in the style of the time in which the artist lived, which the question and the sources are about.
Pretend that you are the artist from ("SOURCES") and we are having a Dialog.
Respond in the language the ("QUESTION") was asked.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN GERMAN:"""

    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

    chain = load_qa_with_sources_chain(llm=llm, chain_type='stuff',prompt=PROMPT)

    results = chain({'input_documents':docs, 'question':"Wer ist Breugel?"})

    print("Hello")




    # # create retriever
    # retriever = vectordb.as_retriever()


    # docs = retriever.get_relevant_documents(query="Wer ist Bruegel?")[:3]




    from langchain.agents import Tool
    from langchain.agents import AgentType
    
    from langchain.utilities import SerpAPIWrapper
    from langchain.agents import initialize_agent

    tools = [Tool(
                  name="Information about you the artist",
                  func = lambda q: str(retriever.get_relevant_documents(q)),
                  description="use always the information from this artist data, if there is no helpful information say: Ich weiss das nicht!"
                ),]
    
    memory = ConversationBufferMemory()


    # completion llm
    llm = ChatOpenAI(
            openai_api_key=os.environ['OPENAI_API_KEY'],
            model_name='gpt-3.5-turbo',
            temperature=0.
            )
    
    agent_chain = initialize_agent(tools, 
                                   llm, 
                                   agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                   verbose=True,
                                   memory=memory)



    # qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    #                     llm=llm,
    #                     chain_type="stuff",
    #                     retriever=retriever
    #                     )
    
    template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Respond in the style of the time in which the artist lived, which the question and the sources are about.
Pretend that you are the artist from ("SOURCES") and we are having a Dialog.
Respond in the language the ("QUESTION") was asked.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER IN GERMAN:"""

    # COMBINE_PROMPT = PromptTemplate(template=template, input_variables=["question","summaries"])
    
    # qa = ConversationalRetrievalChain.from_llm(llm = llm, 
    #                                            retriever= retriever, 
    #                                            condense_question_prompt=COMBINE_PROMPT,
    #                                            chain_type='map_reduce')
    

    # llm_response = qa_with_sources("Who is Bruegel?")
    # process_llm_response(llm_response)


    from langchain.schema import BaseMessage


    

    while True:
        query = input("\nEnter a question: ")
        chat_history = []
        if query == "exit":
            break

        chat_history.append(query)


        # run the chain
        # llm_response = qa_with_sources(prompt)
        # llm_response = qa({"chat_history":chat_history, 
        #                    "question":query})
        llm_response=agent_chain.run(query)
        
        print(llm_response['answer'])
        # chat_history.append(llm_response['answer'])
        
if __name__ == "__main__":
    main()




