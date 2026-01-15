# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# # huggingfaceEndPoing is used while API Inference Endpoint is being called.

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id = "",
#     task = "text-generation"
# )

# model = ChatHuggingFace(llm = llm)

# result = model.invoke("who is Monkey D. Luffy?")

# print(result.content)

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
result = chat_model.invoke("Who is Monkey D. Luffy?")
print(result.content)
