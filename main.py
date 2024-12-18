from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Azure OpenAI configuration
azure_openai_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")  # Your Azure model deployment name
azure_api_base = os.getenv("AZURE_API_BASE")  # Your Azure endpoint
azure_api_key = os.getenv("AZURE_API_KEY")  # Your Azure subscription key
azure_api_version = "2024-05-01-preview"  # Azure API version

# Initialize Azure OpenAI model via LangChain
llm = AzureChatOpenAI(
    azure_deployment=azure_openai_deployment_name,
    openai_api_version=azure_api_version,
    azure_endpoint=azure_api_base,
    openai_api_key=azure_api_key,
    model_name="gpt-4",  # Change if using a different model
)

# Request body schema
class QueryRequest(BaseModel):
    question: str

# Response body schema
class QueryResponse(BaseModel):
    response: str

@app.post("/ask-house", response_model=QueryResponse)
async def ask_house(query: QueryRequest):
    """
    Endpoint to ask Azure OpenAI how to build a house.
    """
    # Construct the prompt
    user_message = HumanMessage(content=query.question)

    # Generate a response
    response = llm.invoke([user_message])

    return QueryResponse(response=response.content)

