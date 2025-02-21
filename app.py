# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st

st.set_page_config(page_title="Mistral-7B ChatBot 🤖", page_icon="🤖", initial_sidebar_state="collapsed")

# Hide Streamlit's default UI elements
st.markdown("""
    <style>
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

sec_key = os.getenv("HF_TOKEN")

# Define the LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    model_kwargs={"max_length": 300},
    huggingfacehub_api_token=sec_key
)

# Memory for conversation history (adjusting to prevent AI hallucinations)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# **Updated Prompt Template**
template =  """You are an AI assistant. Answer only based on actual user inputs.
Do not generate extra questions or repeat past messages.

Conversation history:
{chat_history}

User: {question}
AI:"""

prompt = PromptTemplate(template=template, input_variables=["chat_history", "question"])

# Create the LangChain conversation chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit UI
st.title("Mistral-7B ChatBot 🤖")
st.write("Chat with an AI that remembers your past messages!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Say something...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Show user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder for AI response
    response_placeholder = st.empty()

    # Get AI response
    response = chain.run(question=user_input)
    # **Filter out hallucinated content**
    unwanted_phrases = [
        "User:",  # Removes hallucinated user messages
        "AI:",  # Ensures AI doesn't self-identify mid-response
        "Hello! How can I assist you today?"  # Prevent repeated greetings
    ]

   for phrase in unwanted_phrases:
       response = response.replace(phrase, "").strip()

    # Display AI response
    response_placeholder.markdown(response)

    # Add AI response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
