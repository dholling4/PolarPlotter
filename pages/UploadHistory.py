# import streamlit as st

# # Page title
# st.title("Virtual Race Challenge")

# # Introduction and Description
# st.write(
#     "Welcome to the Virtual Race Challenge! Lace up your shoes, hit the pavement, and join our global community in this exciting virtual race."
# )

# # Race Details
# st.header("Race Details")
# st.write(
#     "Date: [Your Virtual Race Date]\n"
#     "Distance: [Your Virtual Race Distance]\n"
#     "Location: Anywhere, it's virtual!\n"
#     "How to Participate: [Instructions for Participation]\n"
# )

# # Participation Form
# st.subheader("Participate Now!")
# user_name = st.text_input("Your Name:")
# user_email = st.text_input("Your Email:")
# st.button("Join the Race")

# # Medals and Badges
# st.header("Win Medals and Badges")
# st.write(
#     "Complete the race and earn exclusive medals and badges to showcase your achievement. Compete with others and climb the leaderboard!"
# )

# # Prizes
# st.header("Exclusive Prizes")
# st.write(
#     "Top Performers and Participants stand a chance to win exciting prizes:\n"
#     "- Free Premium Subscription to The Digital Athlete\n"
#     "- Exclusive Video Content Access\n"
#     "- Free Gear and Merchandise\n"
# )

# # Leaderboard
# st.header("Leaderboard")
# st.write("Track your progress and see how you compare to other participants.")

# # Terms and Conditions
# st.header("Terms and Conditions")
# st.write("Make sure to read and agree to the terms and conditions before participating.")

# # Footer
# st.markdown(
#     """
#     ---
#     *The Virtual Race Challenge is brought to you by [Your Company Name].*
#     """
# )



# # import streamlit as st
# # from hugchat import hugchat
# # from hugchat.login import Login

# # # # App title
# # # st.set_page_config(page_title="AI Coach 🦾 🤖")

# # # st.write("""### 🦾 🤖 Introducing Whalley, Your Personal AI Coach for Peak Performance! 
         
# # # 🌟 Welcome to the future of athletic training with Whalley, your dedicated AI coach on the Digital Athlete app! 🌟""")
# # # whalley_expander = st.expander("📖 Learn more about Whalley", expanded=False)         

# # # with whalley_expander:
# # #     st.markdown("""Whalley is a state-of-the-art AI coach that uses the latest in natural language processing to provide you with personalized feedback and advice to help you reach your peak performance! 🏃‍♀️ 🏃‍♂️""")
             
# # # # ✨ **Data-Driven Insights**: Whalley utilizes state-of-the-art algorithms to analyze data from your wearables, providing precise insights into your biomechanics and performance metrics.

# # # # 🤖 **Adaptive Training**: No two athletes are the same. Whalley tailors your training regimen based on your unique movement signature, ensuring that every workout is optimized for YOUR success.

# # # # 👟 **Smart Gear Recommendations**: Say goodbye to guesswork! Whalley recommends the best footwear and equipment based on your biomechanics, enhancing comfort, reducing injury risks, and optimizing performance.

# # # # 🔒 **Progress Tracking**: Track your journey effortlessly with Whalley. Monitor your improvements, set new goals, and celebrate achievements with our user-friendly interface.

# # # # 👥 **Include Expert Opinion**: Easily share your progress with coaches, trainers, and physical therapists, fostering collaboration for even better results.

# # # # 💡 **Continuous Learning**: Whalley evolves with you. As you grow, so does your AI coach. Benefit from continuous learning algorithms that adapt to your changing needs.

# # # # 🌐 **The Digital Athlete Community**: Join a thriving community of like-minded individuals. Share experiences, participate in challenges, and motivate each other towards greatness!

# # # # 📱 **Seamless Integration**: The Digital Athlete app seamlessly integrates with your wearables, making it easy for you to focus on your training while Whalley takes care of the rest.

# # # # 🚀 **Unleash Your Potential**: Whether you're a seasoned athlete or just starting, Whalley empowers you to reach new heights. If you can move, Whalley can measure it!

# # # # Ready to embark on your journey to peak performance? Download the Digital Athlete app and let Whalley be your guide to success! 🚀💪""")

# # # Hugging Face Credentials
# # with st.sidebar:
# #     st.title('AI Coach 🦾 🤖')
# #     # if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
# #     #     st.success('HuggingFace Login credentials already provided!', icon='✅')
# #     #     hf_email = st.secrets['EMAIL']
# #     #     hf_pass = st.secrets['PASS']
# #     # else:
# #     #     hf_email = st.text_input('Enter E-mail:', type='password')
# #     #     hf_pass = st.text_input('Enter password:', type='password')
# #     #     if not (hf_email and hf_pass):
# #     #         st.warning('Please enter your credentials!', icon='⚠️')
# #     #     else:
# #     #         st.success('Proceed to entering your prompt message!', icon='👉')
# #     # st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

# # # ## OPEN AI
    
# # # import streamlit as st
# # # from langchain.llms import OpenAI

# # # st.title('🦜🔗 Quickstart App')

# # # openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# # # def generate_response(input_text):
# # #     llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
# # #     st.info(llm(input_text))

# # # with st.form('my_form'):
# # #     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
# # #     submitted = st.form_submit_button('Submit')
# # #     if not openai_api_key.startswith('sk-'):
# # #         st.warning('Please enter your OpenAI API key!', icon='⚠')
# # #     if submitted and openai_api_key.startswith('sk-'):
# # #         generate_response(text)