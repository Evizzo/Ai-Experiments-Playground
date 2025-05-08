# Golubiro Spijuniro Voice Chat

A simple CLI voice chatbot that **acts as "golubiro spijuniro"**, using Google’s Gemini 2.0 Flash model for text generation and ElevenLabs for text-to-speech. Speak into your microphone (or type) and get short, sarcastic, and ironic replies.

## Features

* 🎙️ **Voice input** via SpeechRecognition (Google STT) with fallback to text input
* 🤖 **LLM chat** using `langchain_google_genai` with `gemini-2.0-flash`
* 🔊 **Voice output** via ElevenLabs TTS (`elevenlabs`), streaming with `mpv` fallback
* 🚪 **CLI interface**: run in your terminal, type or speak, say `exit` or `quit` to end

## Requirements

Make sure you have Python 3.8+ installed.

```bash
# Install system dependencies on Linux
sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-pyaudio mpv

# Install Python packages
echo "SpeechRecognition pyaudio python-dotenv langchain-google-genai elevenlabs" > requirements.txt
pip install -r requirements.txt
```

> **Note:** mpv is optional but enables lower-latency streaming playback. Without it, the script falls back to pure-Python playback.

## Configuration

1. Create a `.env` file in the project root:

   ```ini
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```
2. (Optional) If you want voice input, ensure your microphone is recognized by your OS.

## Usage

Run the chatbot in your terminal:

```bash
python main.py
```