from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="hyperbolic",  # set your provider here
    # provider="nebius",
    # provider="together",
)

chat_model = ChatHuggingFace(llm=llm)

st.header('Chat with Nargish.AI')

user_input = st.text_input('Enter your prompt here')

if st.button('Generate Response'):
    result = chat_model.invoke(user_input)
    st.write(result.content)