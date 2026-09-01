import os
import requests
import whisper
import time

from dotenv import load_dotenv



# ENVIRONMENT
load_dotenv()



# CONFIGURATION
WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small"
)

SARVAM_API_KEY = os.getenv(
    "SARVAM_API_KEY"
)

SARVAM_MODEL = "saaras:v3"

SARVAM_STT_URL = (
    "https://api.sarvam.ai/speech-to-text"
)



# WHISPER
_whisper_model = None


def load_whisper_model():
    """Load Whisper model only once."""

    global _whisper_model

    if _whisper_model is None:

        print(
            f"\nLoading Whisper model: "
            f"{WHISPER_MODEL}"
        )

        _whisper_model = whisper.load_model(
            WHISPER_MODEL
        )

        print(
            "Whisper model loaded successfully."
        )

    return _whisper_model


def transcribe_chunk_whisper(
    chunk_path: str
) -> str:
    """Transcribe English audio using local Whisper."""

    model = load_whisper_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        fp16=False
    )

    return result["text"].strip()



# SARVAM
def transcribe_chunk_sarvam(
    chunk_path: str
) -> str:
    """
    Transcribe Hindi/Hinglish audio
    and return English text.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set."
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    data = {
        "model": "saaras:v3",
        "mode": "translate",
        "language_code": "hi-IN"
    }

    with open(chunk_path, "rb") as audio_file:

        files = {
            "file": (
                os.path.basename(chunk_path),
                audio_file,
                "audio/wav"
            )
        }

        for attempt in range(3):

            response = requests.post(
                SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=300
            )

            if response.status_code != 429:
                break

            wait_time = 2 ** attempt

            print(
                f"Rate limited. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

        

    if not response.ok:
        print(
            "Sarvam error:",
            response.status_code,
            response.text
        )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "transcript",
        ""
    ).strip()


# TRANSCRIPTION ROUTER
def transcribe_chunk(
    chunk_path: str,
    language: str
) -> str:
    """
    Select transcription engine.

    english  → Whisper
    hinglish → Sarvam
    """

    language = language.lower().strip()

    if language == "english":

        print("→ Using local Whisper")

        return transcribe_chunk_whisper(
            chunk_path
        )

    elif language == "hinglish":

        print("→ Using Sarvam Saaras v3")

        return transcribe_chunk_sarvam(
            chunk_path
        )

    else:

        raise ValueError(
            f"Unsupported language: {language}\n"
            "Use 'english' or 'hinglish'."
        )



# TRANSCRIBE ALL CHUNKS
def transcribe_all(
    chunks: list[str],
    language: str
) -> str:
    """
    Transcribe all chunks using
    the selected transcription engine.
    """

    if not chunks:
        raise ValueError(
            "No audio chunks were provided."
        )

    transcripts = []

    total = len(chunks)

    print(
        f"\nTranscription mode: {language}"
    )

    print(
        f"Total chunks: {total}\n"
    )

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"Transcribing "
            f"chunk {i}/{total}..."
        )

        text = transcribe_chunk(
            chunk_path=chunk,
            language=language
        )

        if text:

            transcripts.append(text)

    return "\n".join(transcripts)