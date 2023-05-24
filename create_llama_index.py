import argparse
import os
from dotenv import load_dotenv
import yaml

from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper
from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index import load_index_from_storage
from langchain.llms import OpenAI

# Define the command line arguments
parser = argparse.ArgumentParser(description='Create a vector database for the Linux Language Model.')
parser.add_argument('--output-dir', '-o', default='', type=str, help='The directory where the vector database will be stored.')
parser.add_argument('--source-dir', '-s', default='', type=str, help='The directory containing the source files.')
parser.add_argument('--cfg_llama','-c', default='./cfgs/cfg_llama_index.yml')

def main():
    # Parse the command line arguments
    args = parser.parse_args()
    
    #Load the config
    with open(args.cfg_llama, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the API keys
    load_dotenv()

    persist_directory: str = args.output_dir if len(args.output_dir) else os.environ.get('PERSIST_DIRECTORY', 'llama_index')
    source_directory: str = args.source_dir if len(args.source_dir) else os.environ.get('SOURCE_DIRECTORY', 'raw_data')
    
    current_dir = os.getcwd()
    
    path_persist_directory = os.path.join(current_dir,persist_directory)
    
    if not os.path.exists(path_persist_directory):
        # create directory for storing the index
        os.makedirs(path_persist_directory)
    
    node_parser = SimpleNodeParser()
    documents = SimpleDirectoryReader(source_directory).load_data()
    # nodes = node_parser.get_nodes_from_documents(documents)
    
    print("Loading of documents finished!")
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="text-davinci-002"))
    
    prompt_helper = PromptHelper(config['max_input_size'], config['num_output'], config['max_chunk_overlap'])

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    
    print("Index succesfully created!")
    
    
    index.storage_context.persist(persist_dir=path_persist_directory)
    
    print(f"Index succesfully stored to {persist_directory}")
    

if __name__ == "__main__":
    main()