import argparse
import os
from dotenv import load_dotenv
import yaml
from llama_index import LLMPredictor, PromptHelper
from llama_index import ServiceContext, StorageContext
from llama_index import load_index_from_storage
from langchain.llms import OpenAI
from llama_index.langchain_helpers.agents import IndexToolConfig, create_llama_chat_agent, create_llama_agent, LlamaIndexTool
from llama_index.langchain_helpers.agents import LlamaToolkit
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

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
    
    # llm = OpenAI(temperature=0., model_name="text-davinci-002")
    
    llm = OpenAI(temperature=0.)
    
    llm_predictor = LLMPredictor(llm=llm)
    
    persist_directory = args.output_dir if len(args.output_dir) else os.environ['PERSIST_DIRECTORY']
    
    prompt_helper = PromptHelper(config['max_input_size'], config['num_output'], config['max_chunk_overlap'])

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    storage_context = StorageContext.from_defaults(persist_dir=persist_directory)
    
    index = load_index_from_storage(service_context=service_context,
                                    storage_context=storage_context)
    
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    "------------------------------------------------------------------------------------------------------"
    
    tools = [Tool(name="GPT Index",
                func=lambda q: str(index.as_query_engine().query(q)),
                description="useful for awnsering questions about Linux",
                return_direct=True),
            ]
    
    agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)
    
    
    while True:
        prompt = input("User: ")
        if prompt == "exit":
            break
        
        output = agent_executor.run(input=prompt)
        print(output)
    
    
    
    "------------------------------------------------------------------------------------------------------"
    
    # query_engine = index.as_query_engine(retriever_mode='embedding')
    
    # tool_config = IndexToolConfig(query_engine=query_engine, 
    #                                 name=f"Vector Index",
    #                                 description=f"useful for when you want to answer queries about Linux",
    #                                 tool_kwargs={"return_direct": True,
    #                                              "return_sources": True}
    #                             )
    
    # # tool = LlamaIndexTool.from_tool_config(tool_config)
    
    
    
    # toolkit = LlamaToolkit(llm=llm, index_config = tool_config)
    
    # agent_chain = create_llama_chat_agent(toolkit,
    #                                       llm,
    #                                       memory=memory,
    #                                       verbose=True
    #                                     )
    
    # while True:
    #     text_input = input("User: ")
    #     response = agent_chain.run(input=text_input)
    #     print(f'Agent: {response}')


if __name__ == "__main__":
    main()