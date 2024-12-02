from openai import OpenAI 
import streamlit as st
import random
import time
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import datetime

# openai API키 입력
load_dotenv()


client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

# 제목
st.title("77M🤖")

# openai 응답 채팅창 설정
def openai_stream(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            stream=True  # 실시간으로 입력 송출(False면 입력이 완료된 후에 송출)
        )

        # Streamlit 텍스트 출력을 위한 공간
        answer_container = st.empty()

        full_response = ""

        # 응답을 스트리밍으로 하나씩 출력
        for chunk in response:
            if chunk.choices[0].delta.content != None:
                full_response += chunk.choices[0].delta.content
                answer_container.markdown(full_response)  # 실시간으로 출력
        return full_response

    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    return full_response

# 입력창
if prompt := st.chat_input("대한민국의 국보 또는 보물을 입력하시거나 물어보세요"):    
    # 입력 메세지 저장
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 화면에 입력한 메세지 출력(Markdown 형식)
    with st.chat_message("user"):
        st.markdown(prompt)
    # 사용자가 입력한 메세지를 session_state.messages에 저장
    st.session_state.messages.append({"role": "user", "content": prompt})    

    # openai의 응답
    with st.chat_message("assistant"):
        response = openai_stream(prompt) 
    # openai의 응답을 session_state.messages에 저장
    st.session_state.messages.append({"role": "assistant", "content": response})


# 사이드 바 설정

# 지도 설정
data = pd.DataFrame({
    'lat':[37.56],
    'lon':[127],
	})

st.sidebar.title("지도")
with st.sidebar:
    add_radio = st.map(data,latitude='lat', longitude='lon')


# 날짜
st.sidebar.title("달력")
with st.sidebar:
    travel_date = st.date_input("날짜를 클릭 하세요", datetime.date(2024, 12, 1))