import os 

# from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from core.vector_db import load_vector_store , get_retriever
from operator import itemgetter

def get_llm():
    return ChatGoogleGenerativeAI(model = "gemini-3.6-flash", google_api_key = os.getenv("GEMINI_API_KEY"))

def format_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def load_rag_chain():
    """Load and return the RAG question-answering chain."""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Be concise and precise.

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": (
                itemgetter("question")
                | retriever
                | RunnableLambda(format_context)
            ),
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str):
    # print(f"Question: {question}")

    answer = rag_chain.invoke({
        "question": question
    })

    # print("Answer:", answer)
    return answer