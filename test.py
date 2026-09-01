from utils.audio_processor import process_input
from core.Transcribe import transcribe_all


source = (
    "https://www.youtube.com/watch?v=1MaloLDDDUg"
)


language = "hinglish"

# Process audio
chunks = process_input(
    source
)

# Transcribe
transcript = transcribe_all(
    chunks,
    language=language
)

# Result
print("\n" + "=" * 60)
print("FINAL TRANSCRIPTION")
print("=" * 60)
print(transcript)