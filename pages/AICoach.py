import streamlit as st
st.write("coming soon...")

import streamlit as st
import cv2
import tempfile
import numpy as np

def show_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = st.slider("Select frame", 0, total_frames - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx}")
    else:
        st.error("Failed to read the frame.")

video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
if video_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name
    show_frame_from_video(tmp_path)

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