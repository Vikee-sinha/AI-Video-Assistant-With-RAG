from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os
def get_llm():
    return ChatMistralAI(model = "mistal-small-latest" ,mistral_api_key = os.getenv("MISTRAL_API_KEY"))

def split_transcribe(transcript: str)->list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    return splitter.split_text(transcript)

def summerizer(transcript:str)->str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system","summerize the portion of the meeting text conciesly."),
        ("human","{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser

    chunks = split_transcribe(transcript)

    chunk_summerizer = (map_chain.invoke({"text":chunk}) for chunk in chunks )

    combined = "\n\n".join(chunk_summerizer)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert meeting summerizer. combine these parcial summerizer"
         "into the final professional meeting summery in bullet points "
         ),
        ("human","{text}"),
    ])

    combined_chain=(
        RunnablePassthrough |RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm |StrOutputParser
    )
    return combined_chain.invoke(combined)

def title_generate(transcript: str)->str:
    llm = get_llm()

    title_chain=(
        RunnablePassthrough |RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
            ("system","based on the meeting transcript , generate a short profesional meeting title"
             "(max world length 8)only return the title nothing else "),
            ("human","{text}"),
        ])
        | llm
        | StrOutputParser
    )
    return title_chain.invoke(transcript[:2000])

