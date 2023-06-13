from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os

def main():
    # Load the API keys
    load_dotenv()

    # load the vector store
    persist_directory = os.environ['PERSIST_DIRECTORY']
    print(persist_directory)

    embeddings = OpenAIEmbeddings() 

    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()

    #initialize the llm
    llm = OpenAI(temperature=0)

    #initailize llm for answering
    llmPieter = OpenAI(temperature=0.5)

    #llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

    #Load memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True)

    #initialize the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
         
        retriever=retriever, 
        memory=memory, #either memory or source documents....
        #return_source_documents=True 
        verbose=True
    )



    while True:
        query = input("\nWas willst du wissen: ")
        if query == "exit":
            break
        
        #only to show source documents otherwise use memory
        #chat_history = []
        #result = qa({"question": query, "chat_history": chat_history})
        #print(result['source_documents'][0])
        

        #result = qa({"question": query,})
        #print(result["answer"])

        #result = retriever.aget_relevant_documents("Pieter Bruegel")


        #maybe check if docs are neccessary to answer the question
        #-----------------------------------------------------------

        
        searchPromt = ("You are Pieter Bruegel \
                       Use this question of a user and translate it to ENGLISH \
                       then restructure it to the third person view \
                       then restructure the question in a way that is relevant for a similarity search in the documents \
                       documents: \n" + query + "\n\n")
        searchResult = llm.predict(searchPromt)#llm(searchPromt)

        print(searchResult + "\n\n")

        result = vectordb.similarity_search(searchResult)



        relevantInformation = "You are the GERMAN speaking Pieter Bruegel and answer in the first person with information that is provided. \
                               Use the linguistic style of people living in Germany in th 16th century. \
                               \n\n\n  This is your information to respond to the message: \n"
        for r in result:
            relevantInformation += str(r.page_content) + "\n------------------\n"
        
        
        relevantInformation += ("\n\n This is the message you have to respond: \n" + query + "\n\n") 

        print(relevantInformation)

        llm_response = llmPieter(relevantInformation)

        print(llm_response)


        #print("test")
        #print(result)




if __name__ == "__main__":
    main()

