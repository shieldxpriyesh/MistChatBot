# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st

# Streamlit UI Configurations
st.set_page_config(page_title="Mistral-7B ChatBot 🤖", page_icon="🤖", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

#Hugging Face API token
sec_key = os.getenv("HF_TOKEN")

#LLM with Hugging Face API
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    model_kwargs={"max_length": 300},
    huggingfacehub_api_token=sec_key
)

# Memory to store chat history
memory = ConversationBufferMemory(memory_key="chat_history")

#Prompt for chain-of-thought reasoning
template = """You are a conversational AI assistant that responds in a friendly and helpful manner. 
You think step-by-step before answering. Keep responses engaging and relevant to the conversation.

Conversation history:
{chat_history}

User: {question}
AI:"""

prompt = PromptTemplate(template=template, input_variables=["chat_history", "question"])

# LangChain conversation chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit UI
st.title("Mistral-7B ChatBot 🤖")
st.write("Chat with an AI that remembers and thinks step-by-step!")

# Initialize Chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Say something...")
if user_input:
    # user input to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Get AI response
    response = chain.run(question=user_input)
    
    # Add AI response to chat history
    st.session_state["messages"].append({"role": "assistant", "content": response})
    
    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(response)
