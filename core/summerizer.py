import os
import time

import httpx
# from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# The free ("Experiment") tier is capped at roughly 1 request/second per
# model, plus a tokens/minute and tokens/month cap. mistral-small-latest
# fits comfortably in the free tier; mistral-medium/large will 429 much
# faster under the same limits.



def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
    )


def invoke_with_backoff(chain, payload, max_attempts: int = 6, base_delay: float = 2.0):
    """
    Call chain.invoke(payload), retrying on HTTP 429 with exponential
    backoff. Needed because the free Mistral tier allows ~1 req/sec and
    will reject bursts outright rather than queueing them.
    """

    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_attempts:
                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"Rate limited (429). Retrying in {delay:.0f}s "
                    f"(attempt {attempt}/{max_attempts})..."
                )
                time.sleep(delay)
                continue
            raise


def build_analysis_chain():

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting analyst.

Analyze the complete meeting transcript.

Return EXACTLY these five sections:

TITLE:
A short professional meeting title. Maximum 8 words.

SUMMARY:
A concise professional summary in bullet points.

ACTION ITEMS:
List every action item.

KEY DECISIONS:
List all important decisions made during the meeting.

OPEN QUESTIONS:
List all unresolved questions or topics requiring follow-up.

Rules:
- Use ONLY information explicitly present in the transcript.
- Do not invent names, deadlines, decisions, or facts.
- If an owner is not mentioned, write "Not specified".
- If a deadline is not mentioned, write "Not specified".
- If no information exists for a section, write "None found".
- Keep the response concise.
"""
        ),
        (
            "human",
            "{transcript}"
        )
    ])

    return prompt | llm | StrOutputParser()


def analyze_meeting(transcript: str) -> str:

    chain = build_analysis_chain()

    print("\nAnalyzing meeting with Gemini...")

    return invoke_with_backoff(chain, {"transcript": transcript})


def parse_analysis(result: str) -> dict:

    sections = {
        "title": "",
        "summary": "",
        "action_items": "",
        "key_decisions": "",
        "questions": ""
    }

    section_map = {
        "TITLE:": "title",
        "SUMMARY:": "summary",
        "ACTION ITEMS:": "action_items",
        "KEY DECISIONS:": "key_decisions",
        "OPEN QUESTIONS:": "questions"
    }

    current_section = None

    for line in result.splitlines():

        line = line.strip()

        if line in section_map:
            current_section = section_map[line]
            continue

        if current_section:
            sections[current_section] += line + "\n"

    for key in sections:
        sections[key] = sections[key].strip()

    return sections