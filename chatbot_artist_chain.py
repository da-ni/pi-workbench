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

    # create retriever
    retriever = vectordb.as_retriever()

    # completion llm
    llm = ChatOpenAI(
            openai_api_key=os.environ['OPENAI_API_KEY'],
            model_name='gpt-3.5-turbo',
            temperature=0.3
            )



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

    COMBINE_PROMPT = PromptTemplate(template=template, input_variables=["question","summaries"])
    
    qa = ConversationalRetrievalChain.from_llm(llm = llm, 
                                               retriever= retriever, 
                                               condense_question_prompt=COMBINE_PROMPT,
                                               chain_type='map_reduce')
    

    # llm_response = qa_with_sources("Who is Bruegel?")
    # process_llm_response(llm_response)


    
    while True:
        query = input("\nEnter a question: ")
        chat_history = []
        if query == "exit":
            break

        # run the chain
        # llm_response = qa_with_sources(prompt)
        llm_response = qa({"chat_history":chat_history, 
                           "question":query})
        
        print(llm_response['answer'])
        chat_history.append(llm_response['answer'])




if __name__ == "__main__":
    main()