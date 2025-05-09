# 🧠 Multi-Agent Voice Chat with LangGraph

A command-line voice+text chatbot powered by **LangGraph**, orchestrating conversations between you and four chaotic personas:

* **Golubiro** – paranoid government drone disguised as a pigeon  
* **Rick** – drunken tech-savant spewing genius and sarcasm  
* **Morty** – confused teen stumbling through replies  
* **Jerry** – painfully clueless, always trying his best

Uses **Google Gemini** for brains and **ElevenLabs** for voice.

---

## 🎯 How It Works

1. You speak (or type).
2. The system chooses a character to respond based on your message.
3. Sometimes, a second character follows up (e.g. Golubiro after Rick).
4. Voices are spoken using ElevenLabs.

---

### 🧩 LangGraph Flow

#### Nodes (steps)

| Node               | Purpose                                         |
|--------------------|-------------------------------------------------|
| `captureInput`     | Capture user voice or text                     |
| `classifySpeaker`  | Use Gemini to pick a character (e.g. Rick)     |
| `respond`          | Generate a response as that character          |
| `speak`            | Speak the response aloud                       |
| `followUpResponder`| Optional second persona chimes in              |

#### Conditional Logic

After `speak`, the system checks if follow-up is needed:

- Rick → Golubiro
- Morty → Jerry
- Others → no follow-up

---

## 🗣️ Input & Output

- **Voice input** via `speech_recognition` (fallback to text)
- **Voice output** via `ElevenLabs` using `stream()` or `play()`

If ElevenLabs quota is empty, it prints the line instead.

---

## 🛠️ Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://www.langchain.com/)
- Google Gemini (via `langchain_google_genai`)
- [ElevenLabs TTS](https://www.elevenlabs.io/)
- Python 3.10+

---

## 🚀 Run It

```bash
sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-pyaudio mpv
pip install -r requirements.txt
python main.py
```