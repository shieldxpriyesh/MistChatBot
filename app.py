# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
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
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.5,
    model_kwargs={"max_length": 200},
    huggingfacehub_api_token=sec_key
)

# Streamlit UI
st.title("Mistral-7B QAbot 🤖")
st.write("Ask a question and get an AI-generated response.")

user_input = st.text_input("Enter your question:")
if st.button("Get Answer"):
    template = "What is the answer to {question}?"
    prompt = PromptTemplate(template=template, input_variables=["question"])
    final_prompt = prompt.format(question=user_input)
    
    # Get response from model
    response = llm.invoke(final_prompt)
    st.write(response)
