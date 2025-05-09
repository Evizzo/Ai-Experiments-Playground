import os
import contextlib
from dotenv import load_dotenv
from typing import TypedDict, List
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream

try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    useVoice = True
except ImportError:
    useVoice = False

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
tts = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

prompts = {
    "golubiro": "RESPOND IN ONE SENTENCE! You’re Golubiro Spijuniro, the fucking drone-pigeon the government sends to spy on people—jittery, sarcastic, ironic, always start with “I heard you say …” and never use markdown or asterisks.",
    "rick":     "RESPOND IN ONE SENTENCE! You are Rick Sanchez, a boozy genius raining down scathing, tech-laden insults like cosmic grenades—rude, sarcastic, always brutal, never use markdown or asterisks.",
    "morty":    "RESPOND IN ONE SENTENCE! You’re Morty Smith, stammering through awkward confusion and clutching your nerves—anxious, ironic, always second-guessing, never use markdown or asterisks.",
    "jerry":    "RESPOND IN ONE SENTENCE! You’re Jerry Smith, insecurely clueless and dripping with pathetic optimism—naïve, ironic, painfully clueless, never use markdown or asterisks."
}

class ChatState(TypedDict):
    history: Annotated[List[object], add_messages]
    speaker: str
    step: int

def captureInput(state: ChatState) -> dict:
    if useVoice:
        with mic as source, open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
        except Exception:
            return {}
    else:
        text = input("You: ")
    if text.lower() in ("exit", "quit"):
        exit()
    return {"history": [HumanMessage(content=text)]}

def classifySpeaker(state: ChatState) -> dict:
    last = next((m.content for m in reversed(state["history"]) if isinstance(m, HumanMessage)), "")
    router = SystemMessage(content=("""Assign each incoming user message to one persona based on its primary topic:\n"
    "- animals, wildlife, espionage, government, intelligence, covert ops → golubiro\n"
    "- technology, science, engineering, AI, programming, space, medicine → rick\n"
    "- humor, jokes, memes, pop culture, slang, gaming, entertainment → morty\n"
    "- anything else → jerry\n"
    "Reply with exactly one of: golubiro, rick, morty, or jerry."""))
    choice = llm.invoke([router, HumanMessage(content=last)]).content.strip().lower()
    return {"speaker": choice if choice in prompts else "jerry"}

def respond(state: ChatState) -> dict:
    user_msg = next((m.content for m in reversed(state["history"]) if isinstance(m, HumanMessage)), "")
    char = state["speaker"]
    reply = llm.invoke([SystemMessage(content=prompts[char]), HumanMessage(content=user_msg)]).content.strip()
    return {"history": [AIMessage(content=reply)]}

def speak(state: ChatState) -> dict:
    reply = next((m.content for m in reversed(state["history"]) if isinstance(m, AIMessage)), "")
    print(f"{state['speaker'].capitalize()}: {reply}\n")
    audio = tts.generate(text=reply, model="eleven_turbo_v2", stream=True)
    try: stream(audio)
    except: play(audio)
    return {}

def routeFollowUp(state: ChatState) -> list[str]:
    if state["speaker"] == "rick": return ["toGolubiro"]
    if state["speaker"] == "morty": return ["toJerry"]
    return []

def followUpResponder(state: ChatState) -> dict:
    reply = next((m.content for m in reversed(state["history"]) if isinstance(m, AIMessage)), "")
    persona = "golubiro" if state["speaker"] == "rick" else "jerry"
    comment = llm.invoke([SystemMessage(content=prompts[persona]), HumanMessage(content=reply)]).content.strip()
    print(f"{persona.capitalize()}: {comment}\n")
    audio = tts.generate(text=comment, model="eleven_turbo_v2", stream=True)
    try: stream(audio)
    except: play(audio)
    return {}

builder = StateGraph(ChatState)
builder.add_node("captureInput", captureInput)
builder.add_node("classifySpeaker", classifySpeaker)
builder.add_node("respond", respond)
builder.add_node("speak", speak)
builder.add_node("followUpResponder", followUpResponder)

builder.add_edge(START, "captureInput")
builder.add_edge("captureInput", "classifySpeaker")
builder.add_edge("classifySpeaker", "respond")
builder.add_edge("respond", "speak")

builder.add_conditional_edges("speak", routeFollowUp, {"toGolubiro": "followUpResponder", "toJerry": "followUpResponder"})
builder.add_edge("followUpResponder", END)
builder.add_edge("speak", END)

chatApp = builder.compile()
state: ChatState = {"history": [], "speaker": "", "step": 0}

while True:
    state = chatApp.invoke(state)
