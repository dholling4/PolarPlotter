import streamlit as st
st.write("coming soon...")

from openai import OpenAI

YOUR_API_KEY = "INSERT API KEY HERE"

messages = [
    {
        "role": "system",
        "content": (
            "You are an artificial intelligence assistant and you need to "
            "engage in a helpful, detailed, polite conversation with a user."
        ),
    },
    {
        "role": "user",
        "content": (
            "Count to 100, with a comma between each number and no newlines. "
            "E.g., 1, 2, 3, ..."
        ),
    },
]

client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

# demo chat completion without streaming
response = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=messages,
)
print(response)

# demo chat completion with streaming
response_stream = client.chat.completions.create(
    model="mistral-7b-instruct",
    messages=messages,
    stream=True,
)
for response in response_stream:
    print(response)
    
# # from openai import OpenAI
# import streamlit as st
# # import os
# # import sys
# # from dotenv import load_dotenv, dotenv_values
# # load_dotenv()
# # # 
# # import openai

# # openai.api_key = "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"
# # #
# # import os

# # os.environ["OPENAI_API_KEY"] = "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"
# # # initialize the client
# # client = OpenAI(
# #   base_url="https://api-inference.huggingface.co/v1",
# #   api_key=os.environ.get('HUGGINGFACEHUB_API_TOKEN')  #"hf_xxx" # Replace with your token
# # ) 


# st.title("🤖 Whalley: Your AI Virtual Coach")

# st.text("Coming Soon...")
# # import streamlit as st 
# # from langchain.llms import Ollama
# # llm = Ollama(model="llama2-uncensored:latest") # 👈 stef default

# # colA, colB = st.columns([.90, .10])
# # with colA:
# #     prompt = st.text_input("prompt", value="", key="prompt")
# # response = ""
# # with colB:
# #     st.markdown("")
# #     st.markdown("")
# #     if st.button("🙋‍♀️", key="button"):
# #         response = llm.predict(prompt)
# # st.write(response)