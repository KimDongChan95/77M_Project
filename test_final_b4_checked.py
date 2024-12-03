from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import datetime
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import re

# openai API키 입력
load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

model = ChatOpenAI(model="gpt-4o-mini")

# csv 파일 로드.
loader_1 = CSVLoader('sum_sum_Cultural2.csv',encoding='UTF8')
loader_2 = CSVLoader('place.csv',encoding='UTF8')

Tresures = loader_1.load()
Travel = loader_2.load()

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(Tresures)
splits += recursive_text_splitter.split_documents(Travel)


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

if os.path.exists('./db/faiss'):
    vectorstore = FAISS.load_local('./db/faiss', embeddings, allow_dangerous_deserialization=True)
else:
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local('./db/faiss')

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})


# 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please provide detailed, relevant recommendations based on the context provided. Ensure that all recommendations share the same city, district, and road name, and respond only with information from the data provided. Please recommend places in the same city and district. If there are no places in the same district, recommend places in the same city. Be specific and clear in your recommendations Answer in Korean."),
    ("user", "Context: {context}\\n\\nQuestion: {question}. Provide clear and specific recommendations based on the data.")

])




# 디버깅을 위해 만든 클래스
class SimplePassThrough:
    def invoke(self, inputs, **kwargs):
        return inputs

# 프롬프트 클래스
class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        # response_docs 내용을 trim (가독성을 높여줌)
        if isinstance(inputs, list): # inputs가 list인 경우. 즉 여러개의 문서들이 검색되어 리스트로 전달된 경우
            context_text = "\n".join([doc.page_content for doc in inputs]) # \n을 구분자로 넣어서 한 문자열로 합치기
        else:
            context_text = inputs # 리스트가 아닌경우는 그냥 리턴

        # 프롬프트
        formatted_prompt = self.prompt_template.format_messages( # 템플릿의 변수에 삽입
            context=context_text, # {context} 변수에 context_text, 즉 검색된 문서 내용을 삽입
            question=inputs.get("question", "")
        )
        return formatted_prompt

# Retriever 클래스
class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, inputs):
        # 0단계 : query의 타입에 따른 전처리
        if isinstance(inputs, dict): # inputs가 딕셔너리 타입일경우, question 키의 값을 검색 쿼리로 사용
            query = inputs.get("question", "")
        else: # 질문이 문자열로 주어지면, 그대로 검색 쿼리로 사용
            query = inputs
        # 1단계 : query를 리트리버에 넣어주고, response_docs를 얻기
        response_docs = self.retriever.get_relevant_documents(query) # 검색을 수행하고 검색 결과를 response_docs에 저장
        return response_docs

# RAG 체인 설정
rag_chain_debug = {
    "context": RetrieverWrapper(retriever), # 클래스 객체를 생성해서 value로 넣어줌
    "prompt": ContextToPrompt(contextual_prompt),
    "llm": model
}


# 사이드 바 설정

# 지도 설정
data = pd.DataFrame({
    'lat':[37.56],
    'lon':[127],
	})

with st.sidebar.expander("지도 펼치기"):
    st.map(data,latitude='lat', longitude='lon')


# 날짜
# 오늘 날짜 자동 설정
today = datetime.date.today()

# 사이드바에 날짜 입력 위젯을 추가하고, 오늘 날짜를 기본값으로 설정
st.sidebar.title("달력")
travel_date = st.sidebar.date_input("날짜를 클릭 하세요", today)

    
import time
# 진행률 표시
progress_bar = st.progress(0)

# 100번의 반복을 통해 진행률 표시 및 시간을 0.01초로 해서 빠른 속도로 로딩
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i)



# HTML을 사용해 제목을 가운데 정렬
st.markdown("<h1 style='text-align: center;'>77M PROJECT🤖</h1>", unsafe_allow_html=True)


# 이미지
# 이미지 캡션과 함께 가운데 정렬
st.markdown("""
    <style>
        .centered-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# 이미지 출력
st.image("경복궁.jpg", caption="🤖이런 멋진 문화재.. 주변에 무엇이 있을지 궁금하지 않나요?🤖", use_container_width=True)

# 공지글 표시 (투명한 배경과 주황색 글씨)
st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.5); padding: 10px; border-radius: 5px;">
        <strong style="color: orange;">▶</strong> 문화재 주변의 여행지 추천 챗봇입니다!<br>  
        대한민국의 궁금한 국보나 보물을 입력하면 주변 여행지를 추천해드립니다.<br>  
        " 000의 근방 추천해줘 "라고 입력해주세요!
    </div>
""", unsafe_allow_html=True)


# 대화 기록을 키워드별로 저장하는 함수
def store_conversation(keyword, message):
    if keyword not in st.session_state.messages:
        st.session_state.messages[keyword] = []
    st.session_state.messages[keyword].append(message)

# 세션 상태에 메시지 기록이 없으면 초기화
if "messages" not in st.session_state:
    st.session_state.messages = {}


# 사용자가 입력한 질문 처리
if query := st.chat_input("대한민국의 국보 또는 보물을 입력하시거나 물어보세요"):
    with st.chat_message("user"):
        st.markdown(query)
    # 사용자의 질문에서 키워드를 감지, 000이라고 작성해도 되는지 몰라서 일단 변수x라고 작성. "x의 근방 추천해줘"에서 x 가져오기 전에 하는 작업?
    match = re.match(r"(\S+)\s*의\s*근방\s*추천해줘", query)
    if match:
        # 'x'에 해당하는 키워드 가져오기
        keyword = match.group(1)
    # 가져온 키워드에 대해 대화 기록을 저장
        store_conversation(keyword, {"role": "user", "content": query})

    # 1. 리트리버로 사용자의 질문에 대한 검색 결과를 response_docs에 저장
    response_docs = rag_chain_debug["context"].invoke({"question": query})

    # 2. 프롬프트에 질문과 response_docs를 넣어서 응답을 생성
    prompt_messages = rag_chain_debug["prompt"].invoke({
    "context": response_docs,
    "question": query
    })

    # 3. 완성된 프롬프트를 LLM에 넣어 응답을 생성
    response = rag_chain_debug["llm"].invoke(prompt_messages)
    # 가져온 키워드에 대해 어시스턴트의 응답도 저장
    store_conversation(keyword, {"role": "assistant", "content": response.content})

    # 어시스턴트의 응답을 화면에 표시
    with st.chat_message("assistant"):
        if response and hasattr(response, 'content') and response.content:
            st.markdown(response.content)
        else:
            st.markdown("죄송합니다. 관련 정보를 찾을 수 없습니다.")

# 사이드바에 클릭 가능한 키워드를 표시(대화 내용에 맞춰서)
st.sidebar.title("대화 기록")
for keyword in st.session_state.messages.keys():
    if st.sidebar.button(keyword):
        # 키워드를 클릭하면 해당 키워드에 관련된 대화 내역을 표시
            for msg in st.session_state.messages[keyword]:
                st.sidebar.markdown(f"**{msg['role']}**: {msg['content']}")












  
    
    
    
    
