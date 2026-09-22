import ollama

messages = []

print("Chatbot initialized with llama3.2:1b. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = ollama.chat(
        model="llama3.2:1b",
        messages=messages
    )

    bot_reply = response["message"]["content"]

    print(f"\nAI: {bot_reply}\n")

    messages.append({
        "role": "assistant",
        "content": bot_reply
    })