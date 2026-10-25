from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.prompts import get_chat_prompt_template

def get_medical_chain(model_name: str, temperature: float = 0.2):
    """
    Initializes ChatGoogleGenerativeAI and constructs the LCEL RAG chain.
    """
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature,max_retries=1)
    prompt = get_chat_prompt_template()
    
    chain = prompt | llm | StrOutputParser()
    return chain, llm

def stream_narrative(llm, inputs: dict):
    """
    Streams response chunks live using llm.stream() for a typing effect.
    """
    prompt = get_chat_prompt_template()
    messages = prompt.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content