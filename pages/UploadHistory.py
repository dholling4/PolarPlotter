import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

# # App title
# st.set_page_config(page_title="AI Coach 🦾 🤖")

# st.write("""### 🦾 🤖 Introducing Whalley, Your Personal AI Coach for Peak Performance! 
         
# 🌟 Welcome to the future of athletic training with Whalley, your dedicated AI coach on the Digital Athlete app! 🌟""")
# whalley_expander = st.expander("📖 Learn more about Whalley", expanded=False)         

# with whalley_expander:
#     st.markdown("""Whalley is a state-of-the-art AI coach that uses the latest in natural language processing to provide you with personalized feedback and advice to help you reach your peak performance! 🏃‍♀️ 🏃‍♂️""")
             
# # ✨ **Data-Driven Insights**: Whalley utilizes state-of-the-art algorithms to analyze data from your wearables, providing precise insights into your biomechanics and performance metrics.

# # 🤖 **Adaptive Training**: No two athletes are the same. Whalley tailors your training regimen based on your unique movement signature, ensuring that every workout is optimized for YOUR success.

# # 👟 **Smart Gear Recommendations**: Say goodbye to guesswork! Whalley recommends the best footwear and equipment based on your biomechanics, enhancing comfort, reducing injury risks, and optimizing performance.

# # 🔒 **Progress Tracking**: Track your journey effortlessly with Whalley. Monitor your improvements, set new goals, and celebrate achievements with our user-friendly interface.

# # 👥 **Include Expert Opinion**: Easily share your progress with coaches, trainers, and physical therapists, fostering collaboration for even better results.

# # 💡 **Continuous Learning**: Whalley evolves with you. As you grow, so does your AI coach. Benefit from continuous learning algorithms that adapt to your changing needs.

# # 🌐 **The Digital Athlete Community**: Join a thriving community of like-minded individuals. Share experiences, participate in challenges, and motivate each other towards greatness!

# # 📱 **Seamless Integration**: The Digital Athlete app seamlessly integrates with your wearables, making it easy for you to focus on your training while Whalley takes care of the rest.

# # 🚀 **Unleash Your Potential**: Whether you're a seasoned athlete or just starting, Whalley empowers you to reach new heights. If you can move, Whalley can measure it!

# # Ready to embark on your journey to peak performance? Download the Digital Athlete app and let Whalley be your guide to success! 🚀💪""")

# # Hugging Face Credentials
# with st.sidebar:
#     st.title('AI Coach 🦾 🤖')
#     # if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
#     #     st.success('HuggingFace Login credentials already provided!', icon='✅')
#     #     hf_email = st.secrets['EMAIL']
#     #     hf_pass = st.secrets['PASS']
#     # else:
#     #     hf_email = st.text_input('Enter E-mail:', type='password')
#     #     hf_pass = st.text_input('Enter password:', type='password')
#     #     if not (hf_email and hf_pass):
#     #         st.warning('Please enter your credentials!', icon='⚠️')
#     #     else:
#     #         st.success('Proceed to entering your prompt message!', icon='👉')
#     # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
# hf_e = "dholling4@gmail.com"
# hf_p = "John316316!!"

# # hf_pass = st.secrets['PASS']
# # hf_email = st.secrets['EMAIL']
# # Store LLM generated responses
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]


# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # Function for generating LLM response
# def generate_response(prompt_input, email, passwd):
#     # Hugging Face Login
#     sign = Login(email, passwd)
#     cookies = sign.login()
#     # Create ChatBot                        
#     chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
#     return chatbot.chat(prompt_input)

# # User-provided prompt
# if prompt := st.chat_input(disabled=not (hf_e and hf_p)):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)

# # Generate a new response if last message is not from assistant
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = generate_response(prompt, hf_e, hf_p) 
#             st.write(response) 
#     message = {"role": "assistant", "content": response}
#     st.session_state.messages.append(message)



import streamlit as st
from hugchat import hugchat
from hugchat.login import Login

# App title
st.set_page_config(page_title="🤗💬 HugChat")

# Hugging Face Credentials
with st.sidebar:
    st.title('🤗💬 HugChat?')
    if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
        st.success('HuggingFace Login credentials already provided!', icon='✅')
        hf_email = st.secrets['EMAIL']
        hf_pass = st.secrets['PASS']
    else:
        hf_email = st.text_input('Enter E-mail:', type='password')
        hf_pass = st.text_input('Enter password:', type='password')
        if not (hf_email and hf_pass):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot                        
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)

# User-provided prompt
if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, hf_email, hf_pass) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
