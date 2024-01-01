import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
hf_p = 'John316316!!'

# st.set_page_config(page_title="🤗💬 HugChat")

# Hugging Face Credentials
with st.sidebar:
    st.title('AI Coach :mechanical_arm: :robot_face:')
    # if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
    #     st.success('HuggingFace Login credentials already provided!', icon='✅')
    #     hf_email = st.secrets['EMAIL']
    #     hf_pass = st.secrets['PASS']
    # else:
    #     hf_email = st.text_input('Enter E-mail:', type='password')
    #     hf_pass = st.text_input('Enter password:', type='password')
    #     if not (hf_email and hf_pass):
    #         st.warning('Please enter your credentials!', icon='⚠️')
    #     else:
    #         st.success('Proceed to entering your prompt message!', icon='👉')
    # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
intro = """

🌟 Welcome to The Digital Athlete, where cutting-edge technology meets your personal success journey! 🚀 Introducing your AI chat companion, your virtual coach extraordinaire, known as Whalley! 🏆

Greetings, aspiring champion! Whalley is not just your average AI; it's your digital confidant, your data-driven ally, and your secret weapon on the path to athletic greatness. Imagine having a knowledgeable friend who's also a tech-savvy guru, ready to guide you through the highs and lows of your athletic endeavors.

Whalley is here to decode your wearable data, crunch the numbers, and serve up personalized insights that will elevate your performance to new heights. Whether you're chasing records, mastering a new skill, or simply striving for excellence, Whalley is the trusted companion you can rely on.

Get ready to embark on a journey of self-discovery and athletic triumphs, all with the guidance of your digital mentor, Whalley. Unleash the power of technology, ignite your passion, and let the conversation with success begin! 🚀💪✨
"""
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": intro}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

hf_email = 'dholling4@gmail.com'
# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot                        
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)

# User-provided prompt
if prompt := st.chat_input(disabled=not (hf_email and hf_p)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, hf_email, hf_p) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)