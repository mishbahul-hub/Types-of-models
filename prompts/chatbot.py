from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="hyperbolic",  # set your provider here
    # provider="nebius",
    # provider="together",
)
chat_model = ChatHuggingFace(llm=llm)

# chat history
chat_history = []

while True:
    user_input = input("you :")
    chat_history.append(user_input) # adding user input into chat history
    if user_input == "exit":
        print("Exiting chat...")
        break
    else:
        result = chat_model.invoke(user_input)
        chat_history.append(result.content) # adding AI response into chat history
        print("AI: ", result.content)

print("Chat history:", chat_history)        