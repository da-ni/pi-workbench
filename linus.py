import argparse
import os
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import textwrap

# Define the command line arguments
parser = argparse.ArgumentParser(description='Run the Linux Language Model with an example prompt.')
parser.add_argument('--output-dir','-o', default='db', type=str, help='The directory of the vector database.')

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
    persist_directory = args.output_dir if len(args.output_dir) else os.environ['PERSIST_DIRECTORY']
    print(persist_directory)
    embeddings = OpenAIEmbeddings() # type: ignore
    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

    # create retriever
    retriever = vectordb.as_retriever()

    # create chain 
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),  # type: ignore
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )

    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt == "exit":
            break

        # run the chain
        llm_response = qa_chain(prompt)
        process_llm_response(llm_response)


if __name__ == "__main__":
    main()