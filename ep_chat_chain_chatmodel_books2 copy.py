from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory import ChatMessageHistory
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

    chat_history = ""

    llmSimilarty = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    #llm = OpenAI(temperature=0, model='gpt-3.5-turbo')
    llmPieter = ChatOpenAI(temperature=1, model='gpt-3.5-turbo')


    #template for the similarty search
    prepare_prompt = PromptTemplate(
        input_variables=["query", "chat_history"],
        template ="""You are Pieter Bruegel and take the following steps: \n\
                    1 Translate the <latest_question> of the user and to ENGLISH \n\
                    2 Rephrase the question to the first person view \n\
                    3 Use this translated and rephrashed question and the context \
                    from the <previous_conversation> with user and generate a prompt to find relevant documents \n\n \
                    This is the conversation between you and the user so far <previous_conversation>: \n{chat_history}\n\n \
                    
                    This is the latest question from the user<latest_question> : \n{query}\n\n

                    Here is an example:
                    <latest question>:  Wer bist du
                    <ai_response>: Who is Pieter Bruegel"""
    )

    #chain = LLMChain(llm=llm, prompt=prepare_prompt, verbose=True)
    optimize_query_chain = LLMChain(llm=llmSimilarty, prompt=prepare_prompt, verbose=True)

    finalPrompt = PromptTemplate(
        input_variables=["documents", "chat_history", "query"],
        template ="""  You are the GERMAN speaking Pieter Bruegel are haveing a conversation with a user. \
                                Answer in the first person with information that is provided through the DOCUMENTS. \
                                Use the linguistic style of people living in Germany in th 16th century. \
                                \n These are your DOCUMENTS:\n{documents}\n\n

                                \n This is the conversation between you and the user so far: \n{chat_history}\n\n

                                \n This is your next message to respond to: \n{query}\n"""
    )



      #                          
    response_chain = LLMChain(llm=llmPieter, prompt=finalPrompt, verbose=True)
     

    while True:
        query = input("\nWas willst du wissen: ")
        if query == "exit":
            break
        


        #returns dict
        #searchPrompt = chain({"query": query})

        #returns string
        with get_openai_callback() as cb:
            #searchPrompt = optimize_query_chain.run({"query": query, "chat_history": chat_history})
            searchPrompt = optimize_query_chain({"query": query, "chat_history": chat_history})['text']
            print(f"\nCallback: {cb}\n\n")
        
        print("The Promt for similarity search:  " + searchPrompt + "\n\n")

        relevantDocuments = vectordb.similarity_search(searchPrompt, k=3)

        #print(relevantDocuments)


        with get_openai_callback() as cb:
            pieteResponse= response_chain({"documents": relevantDocuments, "chat_history": chat_history, "query": query})
            #pieteResponse= response_chain({"documents": relevantDocuments, "query": query})
            print(f"\nCallback: {cb}\n")

        print("The final prompt:  " + searchPrompt + "\n\n")
        
        
        print(pieteResponse['text'])

        
        chat_history += f"User message: {query} \n"
        chat_history += f"AI_response: {pieteResponse['text']} \n"
        

        #for r in relevantDocuments:
        #    if isinstance(r, Document):
        #        relevantDocuments += str(r.page_content) + "\n------------------\n"


        #print(relevantDocuments)




if __name__ == "__main__":
    main()

