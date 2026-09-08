# AI-Video-Assistant-With-RAG

An AI-powered video assistant that converts meeting/video audio into text and uses Large Language Models and Retrieval-Augmented Generation (RAG) to understand, summarize, and answer questions about the content.

The application can process YouTube videos, transcribe their audio, generate meeting insights, store the transcript in a vector database, and provide context-aware question answering.

---

## 🚀 Project Overview

The **AI Video Assistant with RAG** is designed to turn long-form video or meeting recordings into useful, searchable information.

Instead of watching an entire video, users can:

- Generate a transcript from the video.
- Get a professional meeting summary.
- Generate a meaningful meeting title.
- Extract action items.
- Identify key decisions.
- Find unresolved questions.
- Ask questions about the meeting using RAG.

The project combines **Speech-to-Text, LLMs, Vector Embeddings, ChromaDB, and Retrieval-Augmented Generation** into one pipeline.

---

## ✨ Features

### 🎙️ Speech-to-Text

The application supports transcription of processed audio using:

- **OpenAI Whisper** for English
- **Sarvam AI Saaras v3** for Hindi/Hinglish

The audio is processed into smaller chunks before transcription.

---

### 📝 Meeting Summarization

The transcript is analyzed using an LLM to generate:

- Professional meeting title
- Concise meeting summary
- Action items
- Key decisions
- Open questions

The analysis is designed to use only information present in the transcript and avoid inventing facts.

---

### 🔎 RAG-Based Question Answering

The transcript is converted into smaller text chunks and stored in **ChromaDB**.

When a user asks a question:

    text
User Question
      ↓
Retriever
      ↓
Relevant Transcript Chunks
      ↓
     LLM
      ↓
Context-Aware Answer
