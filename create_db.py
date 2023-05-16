import os
import argparse
from dotenv import load_dotenv

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Define the command line arguments
parser = argparse.ArgumentParser(description='Create a vector database for the Linux Language Model.')
parser.add_argument('--input-dir', default='raw_data', type=str, help='The directory containing the input files.')
parser.add_argument('--output-dir', default='db', type=str, help='The directory where the vector database will be stored.')

def main():
    # Parse the command line arguments
    args = parser.parse_args()
    # Load the API keys
    load_dotenv()

    # Load PDF files from disk
    loader = PyPDFDirectoryLoader(args.input_dir)
    documents = loader.load()

    #splitting pdfs into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # create the vector store
    persist_directory = args.output_dir
    embeddings = OpenAIEmbeddings() # type: ignore
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # persiste the db to disk
    vectordb.persist()

if __name__ == "__main__": 
    main()

