import os
import contextlib
import shutil
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream

try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    use_voice = True
except ImportError:
    print("speech_recognition not installed. Falling back to text input.")
    use_voice = False

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

eleven_key = os.getenv("ELEVENLABS_API_KEY")
if not eleven_key:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in environment")
tts = ElevenLabs(api_key=eleven_key)

print("---\nGolubiro Spijuniro Chat (say 'exit' to quit)\n---")
while True:
    if use_voice:
        print("Listening...")
        with mic as source:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            print(f"You (voice): {user_input}")
        except Exception as e:
            print(f"Voice error: {e}")
            continue
    else:
        user_input = input("You: ")

    if user_input.lower() in ("exit", "quit"):
        print("Spijuniro out!")
        break

    system_prompt = "DONT USE MARKDOWN Act as golubiro spijuniro, say short words and mock the user with sarcasm and irony, be as rude as possible and say 'I heard you say ...':"
    response = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(user_input)
    ])
    reply = response.content.strip()
    print(f"Spijuniro: {reply}\n")

    audio_stream = tts.generate(text=reply, model="eleven_turbo_v2", stream=True)
    if shutil.which("mpv"):
        try:
            stream(audio_stream)
        except Exception:
            play(audio_stream)
    else:
        play(audio_stream)