import os
import contextlib
import shutil
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
tts = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

prompts = {
    "golubiro": SystemMessage(content=(
        "You’re Golubiro Spijuniro, the jittery, sarcastic pigeon spy—"
        "always start with “I heard you say …” and reply in one sentence with no markdown."
    )),
    "rick":     SystemMessage(content=(
        "You are Rick Sanchez, a boozy genius raining down scathing, tech-laden insults—"
        "one sentence, no markdown."
    )),
    "morty":    SystemMessage(content=(
        "You’re Morty Smith, stammering through awkward confusion—"
        "one sentence, no markdown."
    )),
    "jerry":    SystemMessage(content=(
        "You’re Jerry Smith, insecurely clueless and pathetically optimistic—"
        "one sentence, no markdown."
    )),
}

class ChatState(TypedDict):
    history: Annotated[List[object], add_messages]
    speaker: Optional[str]
    step:     int

def captureInput(state: ChatState) -> dict:
    try:
        import speech_recognition as sr
        recognizer, mic = sr.Recognizer(), sr.Microphone()
        with mic as source, open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        print(f"You (voice): {text}")
    except Exception:
        text = input("You: ").strip()
    if text.lower() in ("exit", "quit"):
        print("Goodbye from the multiverse!")
        exit()
    return {"history": [HumanMessage(content=text)]}

def routeToResponder(state: ChatState) -> dict:
    last = state["history"][-1].content
    router = SystemMessage(content=(
        "Route this user message to one persona:"
        "\n- animals/government/spying → golubiro"
        "\n- tech/science → rick"
        "\n- jokes/cringe → morty"
        "\n- else → jerry"
        "\nReply ONLY with golubiro, rick, morty, or jerry."
    ))
    choice = llm.invoke([router, HumanMessage(content=last)]).content.strip().lower()
    return {"speaker": choice if choice in prompts else "jerry"}

def generateResponse(state: ChatState) -> dict:
    user = state["history"][-1].content
    persona = state["speaker"]
    reply = llm.invoke([prompts[persona], HumanMessage(content=user)]).content.strip()
    return {"history": [AIMessage(content=reply)]}

def speak(state: ChatState) -> dict:
    persona = state["speaker"]
    reply = state["history"][-1].content
    print(f"{persona.capitalize()}: {reply}\n")
    audio = tts.generate(text=reply, model="eleven_turbo_v2", stream=True)
    if shutil.which("mpv"):
        try: stream(audio)
        except: play(audio)
    else:
        play(audio)
    return {}

def routeToCommenter(state: ChatState) -> list[str]:
    lastSpeaker = state["speaker"]
    if lastSpeaker == "rick":
        state["speaker"] = "golubiro"
        return ["toComment"]
    if lastSpeaker == "morty":
        state["speaker"] = "jerry"
        return ["toComment"]
    return []

builder = StateGraph(ChatState)
builder.add_node("captureInput",     captureInput)
builder.add_node("routeToResponder", routeToResponder)
builder.add_node("respondMain",      generateResponse)
builder.add_node("speakMain",        speak)
builder.add_node("routeToCommenter", routeToCommenter)
builder.add_node("respondFollowUp",  generateResponse)
builder.add_node("speakFollowUp",    speak)

builder.add_edge(START,               "captureInput")
builder.add_edge("captureInput",    "routeToResponder")
builder.add_edge("routeToResponder","respondMain")
builder.add_edge("respondMain",     "speakMain")

builder.add_conditional_edges(
    "speakMain",
    routeToCommenter,
    {"toComment": "respondFollowUp"}
)
builder.add_edge("respondFollowUp","speakFollowUp")

builder.add_edge("speakMain",       START)
builder.add_edge("speakFollowUp",  START)

chatApp = builder.compile()
state: ChatState = {"history": [], "speaker": None, "step": 0}

while True:
    state = chatApp.invoke(state)
    state["step"] += 1
