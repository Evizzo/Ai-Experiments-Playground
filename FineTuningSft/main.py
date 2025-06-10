import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def loadTestPrompts():
    return [
    ]

def askChatGpt(prompt: str) -> str:
    chat_response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return chat_response.choices[0].message.content

def saveResults(results: list, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def runTests():
    prompts = loadTestPrompts()
    results = []

    for prompt in prompts:
        print(f">>> Prompt: {prompt}")
        response = askChatGpt(prompt)
        print(response, "\n")
        results.append({"prompt": prompt, "response": response})

    saveResults(results, "chatgpt_test_results.jsonl")
    print("Results saved to chatgpt_test_results.jsonl")

if __name__ == "__main__":
    runTests()
