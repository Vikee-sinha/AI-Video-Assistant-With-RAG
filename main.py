from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.Transcribe import transcribe_all
from core.summerizer import analyze_meeting, parse_analysis
from core.vector_db import get_vector_store
from core.rag_engine import load_rag_chain, ask_question


load_dotenv()


def run_pipeline(
    source: str,
    language: str = "english",
    question: str = ""
) -> dict:

    print(
        f"Running pipeline for source: {source} "
        f"and language: {language}"
    )

    # ----------------------------------
    # 1. Process input
    # ----------------------------------
    processed_file_path = process_input(source)

    # ----------------------------------
    # 2. Transcribe
    # ----------------------------------
    transcript = transcribe_all(
        processed_file_path,
        language
    )

    # ----------------------------------
    # 3. ONE MISTRAL CALL
    # ----------------------------------
    analysis = analyze_meeting(transcript)

    meeting_data = parse_analysis(analysis)

    title = meeting_data["title"]
    summary = meeting_data["summary"]
    action_items = meeting_data["action_items"]
    key_decisions = meeting_data["key_decisions"]
    questions = meeting_data["questions"]

    # ----------------------------------
    # 4. Create vector DB
    # ----------------------------------
    print("\nCreating vector store...")

    get_vector_store(transcript)

    # ----------------------------------
    # 5. Load RAG
    # ----------------------------------
    rag_chain = load_rag_chain()

    # ----------------------------------
    # 6. Initial RAG question
    # ----------------------------------
    answer = ask_question(
        rag_chain,
        question
    )

    return {
        "transcript": transcript,
        "summary": summary,
        "title": title,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "questions": questions,
        "answer": answer,
        "rag_chain": rag_chain
    }


if __name__ == "__main__":

    source_file = input(
        "Enter the path or YouTube URL: "
    )

    language = "english"

    question = (
        "What are the key takeaways from the meeting?"
    )

    results = run_pipeline(
        source_file,
        language,
        question
    )

    # ----------------------------------
    # DISPLAY RESULTS
    # ----------------------------------

    print("\n==============================")
    print("TRANSCRIPT")
    print("==============================")
    print(results["transcript"])

    print("\n==============================")
    print("SUMMARY")
    print("==============================")
    print(results["summary"])

    print("\n==============================")
    print("TITLE")
    print("==============================")
    print(results["title"])

    print("\n==============================")
    print("ACTION ITEMS")
    print("==============================")
    print(results["action_items"])

    print("\n==============================")
    print("KEY DECISIONS")
    print("==============================")
    print(results["key_decisions"])

    print("\n==============================")
    print("OPEN QUESTIONS")
    print("==============================")
    print(results["questions"])

    print("\n==============================")
    print("INITIAL ANSWER")
    print("==============================")
    print(results["answer"])

    # ----------------------------------
    # INTERACTIVE RAG
    # ----------------------------------

    rag_chain = results["rag_chain"]

    print("\nChat with your meeting")
    print("Type 'exit' to quit.\n")

    while True:

        user_question = input("Ask a question: ")

        if user_question.lower().strip() == "exit":
            break

        answer = ask_question(
            rag_chain,
            user_question
        )

        print(f"\nAnswer: {answer}\n")