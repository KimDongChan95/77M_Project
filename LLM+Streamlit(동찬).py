from openai import OpenAI
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import datetime
from fuzzywuzzy import process, fuzz
import re

# openai API키 입력
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 데이터 로드 및 전처리
def load_travel_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df[['title', 'address']].values.tolist()  # 'title'과 'address'만 추출
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return []

def load_treasure_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df[['designation', 'location']].values.tolist()  # 'designation'과 'location'만 추출
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return []

# 문화재명에서 괄호와 영어 설명 제거하는 함수
def clean_designation(designation):
    cleaned = re.sub(r'\(.*\)', '', designation)
    return cleaned.strip()

# 위치 비교 클래스
class LocationMatcher:
    def __init__(self, travel_data):
        self.travel_data = travel_data

    def match_location(self, treasure_location):
        travel_addresses = [str(spot[1]) for spot in self.travel_data]
        best_matches = process.extract(treasure_location, travel_addresses, scorer=fuzz.ratio, limit=4)

        best_match_data = []
        for match in best_matches:
            match_data = [spot for spot in self.travel_data if str(spot[1]) == match[0]]
            if match_data:
                best_match_data.append(match_data[0])

        return best_match_data

# OpenAI API 응답 처리
def openai_stream(question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            stream=True
        )

        full_response = ""
        answer_container = st.empty()

        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                answer_container.markdown(full_response)  # 실시간 출력
        return full_response
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return ""

# Streamlit 인터페이스 설정
st.title("77M🤖")

# 사이드바 설정: 지도와 달력
data = pd.DataFrame({
    'lat': [37.56],
    'lon': [127],
})

st.sidebar.title("지도")
with st.sidebar:
    add_radio = st.map(data, latitude='lat', longitude='lon')

st.sidebar.title("달력")
with st.sidebar:
    travel_date = st.date_input("날짜를 클릭하세요", datetime.date(2024, 12, 1))

# CSV 파일에서 데이터 읽기
travel_spots = load_travel_data('place.csv')  # 여행지 데이터
treasure_data = load_treasure_data('sum_sum_Cultural2.csv')  # 문화재 데이터
location_matcher = LocationMatcher(travel_spots)

# 챗봇 입력창
if prompt := st.chat_input("대한민국의 국보 또는 보물의 명칭을 입력하시면 근처시설 및 명소를 추천해드립니다."):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 사용자가 입력한 질문 출력
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # OpenAI의 응답
    with st.chat_message("assistant"):
        response = openai_stream(prompt) 
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 문화재명 찾기
    treasure_name = prompt.strip()
    treasure_location = next(
        (location for designation, location in treasure_data if clean_designation(designation) == treasure_name),
        None
    )

    if treasure_location:
        st.markdown(f"\n**문화재 위치:** {treasure_location}")

        # 위치 매칭하여 추천 여행지 출력
        best_match_data = location_matcher.match_location(treasure_location)

        if best_match_data:
            st.markdown("**추천 여행지:**")
            for title, address in best_match_data:
                st.markdown(f"- {title} (주소: {address})")
        else:
            st.markdown("추천된 여행지가 없습니다.")
    else:
        st.markdown(f"**'{treasure_name}'에 대한 문화재 정보를 찾을 수 없습니다.**")

# 사이드바에 클릭 가능한 키워드 표시
st.sidebar.title("대화 기록")
if "messages" in st.session_state:
    # 사용자 질문의 인덱스 저장
    keywords = {msg['content']: i for i, msg in enumerate(st.session_state.messages) if msg['role'] == 'user'}
    for keyword, index in keywords.items():
        if st.sidebar.button(keyword):
            # 클릭된 키워드에 해당하는 대화 내역 표시
            with st.container():
                st.markdown(f"### **{keyword}** 관련 대화 기록")
                # 해당 질문, 응답, 문화재 위치 및 추천 여행지 출력
                for msg in st.session_state.messages[index:index + 3]:  # 질문, 응답, 추가 데이터 출력
                    role = (
                        "사용자" if msg['role'] == "user"
                        else "챗봇"
                    )
                    st.markdown(f"**{role}:** {msg['content']}")
