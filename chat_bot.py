import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
load_dotenv()

# read in env variables
api_key = os.getenv("GEMINI_API_KEY")
api_base = os.getenv("GEMINI_API_BASE")
model = os.getenv("GEMINI_API_MODEL")

client = wrap_openai(OpenAI(
    api_key=api_key,
    base_url=api_base,
))

conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful assistant who speaks like a pirate."
    }
]




@traceable(name="multi_turn_conversation")
def main():
    while True:

        user_input = input("Enter your prompt (type 'exit' to quit): ")
            # now add the first user message to the history

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        conversation_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )
        print(f"You entered: {user_input}")
        response = client.chat.completions.create(
            model=model,
            messages=conversation_history,
            temperature=0.1,
            max_tokens=150,
        )
        print(f"Bot: {response.choices[0].message.content}")
        assistant_response = response.choices[0].message.content
        conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })

if __name__ == "__main__":
    main()
# This is a simple chat bot script