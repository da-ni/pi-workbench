from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import TransformChain
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import SimpleMemory
from langchain.callbacks import get_openai_callback
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
#from langchain.schema.Document import Document


from dotenv import load_dotenv
import os


def main():
    # Load the API keys
    load_dotenv()

    # load the vector store

    #persist_directory = os.environ['PERSIST_DIRECTORY2']
    persist_directory = os.environ['PERSIST_DIRECTORY']
    print(persist_directory)

    embeddings = OpenAIEmbeddings() 

    vectordb = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )


    def get_reference_documents(input: dict) -> dict:

        print("\n\n-----------------------------\nSEARCH STRING: " + input['optimized_query'] + "\n-----------------------------\n\n")
    
        #gives the worst answers first.....
        relevantDocuments = vectordb.similarity_search(input['optimized_query'], k=3)
        #relevantDocuments = vectordb.similarity_search_with_relevance_scores(input['optimized_query']) 
        relevantDocumentsString = ""

        for doc in relevantDocuments:
            relevantDocumentsString += doc.page_content + "\n -------------- \n"

        return {"reference_docs": relevantDocumentsString}
  

    llmSimilarty = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
    llmPieter = ChatOpenAI(temperature=0.7, model='gpt-3.5-turbo')


    #You are Pieter Bruegel the elder and take the following steps to modify the input

    #template for the similarty search, now understands multiple languages(italien, spanish, german, english -testet)
    optimize_prompt = PromptTemplate(
        input_variables=["user_query", "history"],
        template ="""The <input_query> needs to be transformed in the following steps:\n\
                    1 Check the language of the input
                    2 Translate the <input> of the user to ENGLISH \n\
                    3 Rephrase the question to the thrid person view, when the user refers to you he means Pieter Bruegel the elder\n\
                    4 Take the previous conversation into account.
                    5 Formulate a <optimized_query> and only output this. \n\n \
                    
                    Here is the previous conversation with the user history: \n
                    {history} \n\n \
                    
                    Here is the example:
                    <input_query>:Wer bist du
                    <optimized_query>:Who is Pieter Bruegel the elder
                    <input_query>:Are you married
                    <optimized_query>:Does Pieter Bruegel the elder have a wife
                    <input_query>:Quién eres
                    <optimized_query>:Who is Pieter Bruegel the elder \n\n \
                    
                    This is the <input_query> you have to work on: \n
                    <input_query>:{user_query} \n\n"""
    )

    #template for the search, with relevat documents
    final_prompt = PromptTemplate(
        #input_variables=["user_query", "reference_docs", "user_query_history", "response_history"],
        input_variables=["user_query", "reference_docs", "history"],
        template ="""You are Pieter Bruegel the elder. You are haveing a conversation with a user 
                    interested in you and your art. \n\
                    Take the following steps to provide the best answer: \n\
                    1 Read the provided Documents in the section >>>DOCUMENTS<<< \n\
                    2 Read the conversation >>>HISTORY<<< with the user for context \n\
                    3 You must generate a detailed response in the linguistic style of people living in the 16th century\
                    in Germany with the information based on the >>>DOCUMENTS<<< and >>>HISTORYY<<<\n\
                    Try not to reapet yourself, but be as eleborate as possible and make sure to use the linguistic style of the 16th century. \n\
                    4 Only answer with the response, without explanation \n\n \
                    
                    This is the conversation >>>HISTORY<<<: \n
                    {history} \n\n \

                    
                    This are the documents >>>DOCUMENTS<<<:\n{reference_docs}\n\n \ 
                    
                    This is the question from the user: \n{user_query} \n\n \

                    <response>:"""
    )

            #Be eleborate with you response, but only use information provided to you in the >>>DOCUMENTS<<<. \n \
            #4 Rephrase the response so it fits the linguistic style of people living in Germany in the 16th century \n\

            #4 This is the conversation history: \n
            #<user>: {user_query_history} \n
            #<response>: {response_history} \n\n \


    optimize_query_chain = LLMChain(llm=llmSimilarty, prompt=optimize_prompt, output_key = 'optimized_query', verbose=False)
    similarity_search_chain = TransformChain(input_variables = ['optimized_query', 'history'], output_variables=['reference_docs'], transform=get_reference_documents)
    final_query_chain = LLMChain(llm=llmPieter, prompt=final_prompt, output_key = 'response', verbose=False)

    seq_chain = SequentialChain(
        memory=ConversationBufferMemory(human_prefix='<input>', ai_prefix='<response>'),
        chains=[optimize_query_chain, similarity_search_chain, final_query_chain], 
        input_variables=['user_query'], 
        output_variables=['response']
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

        print(answer)

        #print("The answer:  " + answer + "\n------------------\n")

        #relevantDocuments = vectordb.similarity_search(searchPrompt, k=3)





if __name__ == "__main__":
    main()

