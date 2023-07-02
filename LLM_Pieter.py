from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import TransformChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferWindowMemory


from dotenv import load_dotenv
import os

class LLM_Pieter:
    def __init__(self, sim_temp = 0, pieter_temp = 0.2):
        
        #load environment variables
        load_dotenv()
        
        #set up the vector store and llms as gloabl variable
        self.optimized_prompt = None
        self.final_prompt = None
        self.seq_chain = None
        self.embeddings = OpenAIEmbeddings() # type: ignore
        self.vectordb = self._initialize_vectordb()
        self.llmSimilarty = ChatOpenAI(temperature=sim_temp, model='gpt-3.5-turbo') # type: ignore
        self.llmPieter = ChatOpenAI(temperature=pieter_temp, model='gpt-3.5-turbo') # type: ignore
        self._load_prompt_templates()
        self._set_up_llm_chains()


    def _initialize_vectordb(self):
        #load the vector store
        persist_directory = os.environ['PERSIST_DIRECTORY']
        print(persist_directory)

        vectordb = Chroma(
            persist_directory=persist_directory, 
            embedding_function=self.embeddings
        )

        return vectordb

        #test the vector store
        #relevantDocuments = self.vectordb.similarity_search("Jaeger im Schnee", k=3)
        #print (relevantDocuments)


    def _load_prompt_templates(self):
        with open(os.environ['OPTIMIZE_PROMPT_FILE'], 'r') as f:
            optimze_prompt_template = f.read()
        
        with open(os.environ['FINAL_PROMPT_FILE'], 'r') as f:
            final_prompt_template = f.read()


        #template for the similarty search, now understands multiple languages(italien, spanish, german, english -testet)
        self.optimized_prompt = PromptTemplate(
            input_variables=["user_query", "history"],
            template = optimze_prompt_template
        )
        
        self.final_prompt = PromptTemplate(      
            input_variables=["user_query", "reference_docs", "history"],
            template = final_prompt_template
        )


    def _set_up_llm_chains(self):
        optimize_query_chain = LLMChain(llm=self.llmSimilarty, prompt=self.optimized_prompt, output_key = 'optimized_query', verbose=True) # type: ignore
        similarity_search_chain = TransformChain(input_variables = ['optimized_query', 'history'], output_variables=['reference_docs'], transform=self._get_reference_documents)
        final_query_chain = LLMChain(llm=self.llmPieter, prompt=self.final_prompt, output_key = 'response', verbose=True) # type: ignore

        #fit the chains together
        self.seq_chain = SequentialChain(
            memory=ConversationBufferWindowMemory(human_prefix='<input>', ai_prefix='<response>', k=3),
            chains=[optimize_query_chain, similarity_search_chain, final_query_chain], 
            input_variables=['user_query'], 
            output_variables=['response']
        )


    def _get_reference_documents(self, input: dict) -> dict:
        print("\n\n-----------------------------\nSEARCH STRING: " + input['optimized_query'] + "\n-----------------------------\n\n")

        relevantDocuments = self.vectordb.similarity_search(input['optimized_query'], k=3)
        relevantDocumentsString = ""

        for doc in relevantDocuments:
            relevantDocumentsString += doc.page_content + "\n -------------- \n"

        #print(relevantDocumentsString)

        return {"reference_docs": relevantDocumentsString}



    def run_query(self, user_input: str):
        print("-----run query-----")
        #response = self.seq_chain.run({"user_query": user_input})

        #see used tokens and price
        with get_openai_callback() as cb:
            response = self.seq_chain.run(user_input)
        print(cb)

        return response

