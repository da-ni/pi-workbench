import argparse
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import textwrap

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

class PromptResponse(BaseModel):
    result: str
    sources: list

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    wrapped_result = wrap_text_preserve_newlines(llm_response['result'])
    sources = [source.metadata['source'] for source in llm_response["source_documents"]]
    return PromptResponse(result=wrapped_result, sources=sources)

@app.post("/api/prompt", response_model=PromptResponse)
def run_prompt(request: PromptRequest):
    prompt = request.prompt

    # Run the chain
    llm_response = qa_chain(prompt)
    processed_response = process_llm_response(llm_response)

    return processed_response

if __name__ == "__main__":
    # Load the API keys and other environment variables
    load_dotenv()

    # Load the vector store
    persist_directory = os.environ['PERSIST_DIRECTORY']
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Create the retriever
    retriever = vectordb.as_retriever()

    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    @app.get("/api/docs", response_class=HTMLResponse)
    async def custom_docs():
        return get_redoc_html(openapi_url="/api/openapi.json", title="API Documentation")

    @app.get("/api/openapi.json")
    async def get_open_api_endpoint():
        return get_openapi(
            title="API Documentation",
            version="0.0.1",
            description="This is an API for running prompts and retrieving results.",
            routes=app.routes,
        )

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)