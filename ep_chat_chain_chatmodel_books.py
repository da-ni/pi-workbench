from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
#from langchain.schema.Document import Document



from dotenv import load_dotenv
import os



def main():
    # Load the API keys
    load_dotenv()

    # load the vector store
    persist_directory = os.environ['PERSIST_DIRECTORY2']
    print(persist_directory)

    embeddings = OpenAIEmbeddings() 

    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    #initialize the llm
    #llm = OpenAI(temperature=0)

    #initailize llm for answering
    #llmPieter = OpenAI(temperature=0.5)

    # not working with model='gpt-3.5-turbo'
    llmSimilarty = OpenAI(temperature=0)
    #llm = OpenAI(temperature=0, model='gpt-3.5-turbo')
    llmPieter = ChatOpenAI(temperature=1, model='gpt-3.5-turbo')


    #template for the similarty search
    prepare_prompt = PromptTemplate(
        input_variables=["query"],
        template ="""  You are Pieter Bruegel. You must only reply the final translated and rephrased question \
                       Use this question of a user and translate it to ENGLISH \
                       then restructure it to the third person view \
                       then rephrase the question in a way that is relevant for a similarity search in the documents \
                       This is the question from the user: {query} + \n\n \
                       Only reply Rephrased for similarity search: <response>"""
    )

    #chain = LLMChain(llm=llm, prompt=prepare_prompt, verbose=True)
    optimize_query_chain = LLMChain(llm=llmSimilarty, prompt=prepare_prompt, verbose=True)

    finalPrompt = PromptTemplate(
        input_variables=["documents", "query"],
        template ="""  You are the GERMAN speaking Pieter Bruegel and answer in the first person with information that is provided. \
                                Use the linguistic style of people living in Germany in th 16th century. \
                                \n These are your documents:\n{documents} \n

                                \n\n\n  This is your information to respond to the message: {query}\n"""
    )

    response_chain = LLMChain(llm=llmPieter, prompt=finalPrompt, verbose=True)
     
                    



    while True:
        query = input("\nWas willst du wissen: ")
        if query == "exit":
            break
        


        #returns dict
        #searchPrompt = chain({"query": query})

        #returns string
        with get_openai_callback() as cb:
            searchPrompt = optimize_query_chain.run({"query": query})
            print(f"\nCallback: {cb}\n\n")
        print(searchPrompt)

        relevantDocuments = vectordb.similarity_search(searchPrompt, k=3)

        print(relevantDocuments)


        with get_openai_callback() as cb:
            pieteResponse= response_chain({"documents": relevantDocuments, "query": query})
            print(f"\nCallback: {cb}\n")
        print(pieteResponse['text'])

        

        #for r in relevantDocuments:
        #    if isinstance(r, Document):
        #        relevantDocuments += str(r.page_content) + "\n------------------\n"


        #print(relevantDocuments)




if __name__ == "__main__":
    main()

