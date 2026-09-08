from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.Transcribe import transcribe_all
from core.summerizer import analyze_meeting, parse_analysis


# Load environment variables
load_dotenv(override=True)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

source = "https://www.youtube.com/watch?v=7aEAS5E5vjg"
language = "english"


# --------------------------------------------------
# 1. PROCESS AUDIO
# --------------------------------------------------

print("\n" + "=" * 60)
print("PROCESSING AUDIO")
print("=" * 60)

chunks = process_input(source)

print(f"Total audio chunks: {len(chunks)}")


# --------------------------------------------------
# 2. TRANSCRIBE
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRANSCRIBING")
print("=" * 60)

transcript = transcribe_all(
    chunks,
    language=language
)


# --------------------------------------------------
# 3. SHOW TRANSCRIPT
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TRANSCRIPTION")
print("=" * 60)

print(transcript[:3000])


# --------------------------------------------------
# 4. ANALYZE COMPLETE MEETING
#    ONE MISTRAL API CALL
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANALYZING MEETING")
print("=" * 60)

result = analyze_meeting(transcript)


# --------------------------------------------------
# 5. PARSE MISTRAL RESPONSE
# --------------------------------------------------

meeting_data = parse_analysis(result)


# # --------------------------------------------------
# # 6. DISPLAY TITLE
# # --------------------------------------------------

# print("\n" + "=" * 60)
# print("TITLE")
# print("=" * 60)

# print(meeting_data["title"])


# # --------------------------------------------------
# # 7. DISPLAY SUMMARY
# # --------------------------------------------------

# print("\n" + "=" * 60)
# print("SUMMARY")
# print("=" * 60)

# print(meeting_data["summary"])


# # --------------------------------------------------
# # 8. DISPLAY ACTION ITEMS
# # --------------------------------------------------

# print("\n" + "=" * 60)
# print("ACTION ITEMS")
# print("=" * 60)

# print(meeting_data["action_items"])


# # --------------------------------------------------
# # 9. DISPLAY KEY DECISIONS
# # --------------------------------------------------

# print("\n" + "=" * 60)
# print("KEY DECISIONS")
# print("=" * 60)

# print(meeting_data["key_decisions"])


# # --------------------------------------------------
# # 10. DISPLAY OPEN QUESTIONS
# # --------------------------------------------------

# print("\n" + "=" * 60)
# print("OPEN QUESTIONS")
# print("=" * 60)

# print(meeting_data["questions"])


# # --------------------------------------------------
# # 11. RAW RESPONSE (OPTIONAL DEBUG)
# # --------------------------------------------------

# print("\n" + "=" * 60)
# print("RAW MISTRAL RESPONSE")
# print("=" * 60)

# print(result)


# # --------------------------------------------------
# # DONE
# # --------------------------------------------------

# print("\n" + "=" * 60)
# print("TEST COMPLETED")
# print("=" * 60)