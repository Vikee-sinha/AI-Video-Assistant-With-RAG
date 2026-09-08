import os
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "Downloads"

os.makedirs(
    DOWNLOAD_DIR,
    exist_ok=True
)


def download_youtube_audio(url: str) -> str:
    """Download audio from a single YouTube video."""

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",

        # Download only the provided video
        "noplaylist": True,

        "outtmpl": output_path,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        filename = ydl.prepare_filename(info)

        # Replace original extension with WAV
        filename = (
            os.path.splitext(filename)[0]
            + ".wav"
        )

        return filename


def convert_to_wav(input_path: str) -> str:
    """Convert audio to mono 16kHz WAV."""

    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )

    audio = AudioSegment.from_file(
        input_path
    )

    # Whisper / speech recognition friendly format
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(
        output_path,
        format="wav"
    )

    return output_path


def chunk_audio(
    wav_path: str,
    chunk_seconds: int = 120
) -> list[str]:
    """
    Split WAV audio into 120-second chunks.

    Sarvam's synchronous STT endpoint
    works with short audio chunks.
    """

    audio = AudioSegment.from_wav(
        wav_path
    )

    chunk_ms = chunk_seconds * 1000

    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms),
        start=1
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = os.path.join(
            DOWNLOAD_DIR,
            f"chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    return chunks


def process_input(url: str) -> list[str]:
    """
    Complete YouTube audio processing pipeline.

    YouTube
        ↓
    Download
        ↓
    Convert to 16kHz mono WAV
        ↓
    Split into 120-second chunks
    """

    print("\nDownloading audio...")

    downloaded_file = download_youtube_audio(
        url
    )

    print(
        f"Downloaded: {downloaded_file}"
    )

    print("\nConverting audio...")

    converted_file = convert_to_wav(
        downloaded_file
    )

    print(
        f"Converted: {converted_file}"
    )

    print("\nCreating audio chunks...")

    chunks = chunk_audio(
        converted_file,
        chunk_seconds=120
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks