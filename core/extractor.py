"""
Alternative Modular Extractors

These functions provide granular extraction if you prefer separate LLM calls
instead of the all-in-one analysis in summerizer.py.
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
    )

def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting analyst.

From the meeting transcript, extract all action items.

For each action item provide:
- Task description
- Owner (who is responsible)
- Deadline (if mentioned, otherwise write 'Not specified')

Format the result as a numbered list.

Do not invent information.
Only use information explicitly present in the transcript.

If no action items are found, say:
'No action items found.'
"""
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting analyst.

From the meeting transcript, extract all key decisions that were made.

Format the result as a numbered list.

Do not invent information.
Only include decisions explicitly mentioned in the transcript.

If no key decisions are found, say:
'No key decisions found.'
"""
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting analyst.

From the meeting transcript, extract all unresolved questions
or topics that need follow-up.

Format the result as a numbered list.

Do not invent information.
Only use information explicitly present in the transcript.

If no unresolved questions are found, say:
'No open questions found.'
"""
    )

    return chain.invoke(transcript)