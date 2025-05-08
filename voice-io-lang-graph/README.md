# 🧠 Multi-Agent Voice Chat with LangGraph

A command-line voice+text chatbot using **LangGraph** to orchestrate a multi-character conversation between you and four AI personas. Characters include:

* **Golubiro** – paranoid pigeon spy
* **Rick** – tech-drunk genius
* **Morty** – awkward teen
* **Jerry** – painfully optimistic dad

Built using **Gemini (Google)** for language, **ElevenLabs** for voice, and **LangGraph** for flow control.

---

## 🎯 How It Works

You talk. The system chooses a responder. Sometimes, a second character chimes in.

### LangGraph Overview

LangGraph acts as a **declarative flow engine**, handling state, routing, and branching logic.

#### 🧩 Nodes

Each node is a **pure function** that takes `ChatState` and returns a partial update:

| Node               | Purpose                                 |
| ------------------ | --------------------------------------- |
| `captureInput`     | Listen or read user input               |
| `routeToResponder` | Use LLM to pick the primary persona     |
| `respondMain`      | Have the chosen persona respond         |
| `speakMain`        | Speak the response aloud                |
| `routeToCommenter` | Rule-based choice for follow-up speaker |
| `respondFollowUp`  | Secondary persona responds (if needed)  |
| `speakFollowUp`    | Speak the follow-up aloud               |

#### 🔁 Edges

LangGraph connects these nodes via:

* **Sequential edges** (e.g. input → classify → respond)
* **Conditional edges** (e.g. only follow up if Rick or Morty spoke)
* **Loopback** (after every full exchange, return to input)

This makes control flow explicit and maintainable—no nested `if` blocks or manual step tracking.

---

## 🗣️ Input & Output

* **Voice input** via `speech_recognition`, fallback to text.
* **Voice output** via ElevenLabs TTS, with `mpv` streaming fallback.

---

## 🧪 Follow-up Logic

Only some personas trigger second replies:

* If **Rick** replies → **Golubiro** always follows up.
* If **Morty** replies → **Jerry** follows up.
* Otherwise, your turn comes next.

This is handled by a **LangGraph conditional edge** routing through `routeToCommenter`.

---

## 🛠️ Tech Stack

* [LangGraph](https://github.com/langchain-ai/langgraph)
* [LangChain](https://www.langchain.com/)
* Google Gemini API (via `langchain_google_genai`)
* [ElevenLabs](https://www.elevenlabs.io/) TTS
* Python 3.10+

---

## 🚀 Run It

```bash
sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-pyaudio mpv
pip install -r requirements.txt
python main.py
```

---
