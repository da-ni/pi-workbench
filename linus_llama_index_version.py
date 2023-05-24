import argparse
import os
from dotenv import load_dotenv
import yaml
from llama_index import LLMPredictor, PromptHelper
from llama_index import ServiceContext, StorageContext
from llama_index import load_index_from_storage
from langchain.llms import OpenAI

# Define the command line arguments
parser = argparse.ArgumentParser(description='Run the Linux Language Model with an example prompt.')
parser.add_argument('--output-dir','-o', default='llama_index', type=str, help='The directory of the vector database.')
parser.add_argument('--cfg_llama','-c', default='./cfgs/cfg_llama_index.yml')


def main():
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Load the config
    with open(args.cfg_llama, 'r') as f:
        config = yaml.safe_load(f)
        
    load_dotenv()
    
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-002"))
    
    persist_directory = args.output_dir if len(args.output_dir) else os.environ['PERSIST_DIRECTORY']
    
    prompt_helper = PromptHelper(config['max_input_size'], config['num_output'], config['max_chunk_overlap'])

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
    
    index = load_index_from_storage(service_context=service_context,
                                    storage_context=storage_context)
    
    query_engine = index.as_query_engine()
    
    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt == "exit":
            break

        # run the chain
        llm_response = query_engine.query(prompt)
        print(llm_response)



if __name__ == "__main__":
    main()