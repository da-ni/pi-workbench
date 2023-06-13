from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import TransformChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.memory import ChatMessageHistory
#from langchain.schema.Document import Document



from dotenv import load_dotenv
import os


def get_reference_documents(input: dict) -> dict:
    persist_directory = os.environ['PERSIST_DIRECTORY2']
    print(persist_directory)

    embeddings = OpenAIEmbeddings() 

    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

    relevantDocuments = vectordb.similarity_search(input['optimized_query'], k=3)

    relevantDocumentsString = ""

    for doc in relevantDocuments:
        relevantDocumentsString += doc.page_content + "\n -------------- \n"

    return {"reference_docs": relevantDocumentsString}



def main():
    # Load the API keys
    load_dotenv()

    # load the vector store
  

    llmSimilarty = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    llmPieter = ChatOpenAI(temperature=1, model='gpt-3.5-turbo')


    #template for the similarty search, now understands multiple languages(italien, spanish, german, english -testet)
    optimize_prompt = PromptTemplate(
        input_variables=["user_query"],
        template ="""You are Pieter Bruegel and take the following steps to modify the input: \n\
                    1 Check the language of the input
                    2 Translate the <input> of the user to ENGLISH \n\
                    3 Rephrase the question to the thrid person view \n\
                    4 Only answer with the response, without explanation \n\n \
                    
                    Here is an example:
                    <input>:Wer bist du
                    <response>:Who is Pieter Bruegel
                    <input>:Are you married
                    <response>:Does Pieter Bruegel have a wife
                    <input>:Quién eres
                    <response>:Who is Pieter Bruegel
                    <input>:{user_query} 
                    <response>:"""
    )

    #template for the search, with relevat documents
    final_prompt = PromptTemplate(
        input_variables=["user_query", "reference_docs"],
        template ="""You are Pieter Bruegel that is haveing a conversation with a user and take the following steps to provide the best answer : \n\
                    1 Read the provided Documents in the section >>>DOCUMENTS<<< \n\
                    2 Generate a response to the user based ton the information in the >>>DOCUMENTS<<< \n\
                    3 Only answer with the response, without explanation \n\n \
                    
                    This are the documents >>>DOCUMENTS<<<: \n {reference_docs} \n\n \ 
                    
                    This is the question from the user: \n {user_query} \n\n \

                    <response>:"""
    )


    optimize_query_chain = LLMChain(llm=llmSimilarty, prompt=optimize_prompt, output_key = 'optimized_query', verbose=False)
    similarity_search_chain = TransformChain(input_variables = ['optimized_query'], output_variables=['reference_docs'], transform=get_reference_documents)
    #final_query_chain = LLMChain(llm=llmPieter, prompt=final_prompt, output_key = 'response', verbose=False)

    seq_chain = SequentialChain(
        chains=[optimize_query_chain, similarity_search_chain], 
        input_variables=['user_query'], 
        output_variables=['reference_docs']
        )


    while True:
        query = input("\nWas willst du wissen: ")
        if query == "exit":
            break
        


        with get_openai_callback() as cb:
            #searchPrompt = optimize_query_chain.run({"query": query, "chat_history": chat_history})
            #searchPrompt = optimize_query_chain.run({"user_query": query})
            #relevantDocs = similarity_search_chain({"optimized_query": searchPrompt})

            answer = seq_chain.run({"user_query": query})

            print(f"\nCallback: {cb}\n\n")
        
        #print("The Promt for similarity search:  " + searchPrompt['text'] + "\n\n")
        #print("The Promt for similarity search:  " + searchPrompt + "\n ------------------\n")
        #print("The documents for similarity search:  " + relevantDocs + "\n------------------\n")

        print(answer)

        #print("The answer:  " + answer + "\n------------------\n")

        #relevantDocuments = vectordb.similarity_search(searchPrompt, k=3)





if __name__ == "__main__":
    main()

