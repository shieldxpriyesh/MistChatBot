# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st
#from google.colab import userdata
st.set_page_config(page_title="Mistral-7B QAbot 🤖", page_icon="🤖", initial_sidebar_state="collapsed")

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

sec_key = os.getenv("HF_TOKEN")

# Define model parameters

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    model_kwargs={"max_length": 200},
    huggingfacehub_api_token=sec_key
)
# Memory to store chat history
memory = ConversationBufferMemory(memory_key="chat_history")

# Define prompt template
template = """You are an advanced AI that thinks step-by-step. Answer the user's question thoughtfully.

Previous conversation:
{chat_history}

New Question: {question}
Step-by-step reasoning:
"""
prompt = PromptTemplate(template=template, input_variables=["chat_history", "question"])

# Create a LangChain conversation chain
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
# Streamlit UI
st.title("Mistral-7B QAbot 🤖")
st.write("Ask a question and get an AI-generated response.")

user_input = st.text_input("Enter your question:")
if st.button("Get Answer"):
    response = chain.run(question=user_input)
    st.write(response)
