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


# openai API키 입력
load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

model = ChatOpenAI(model="gpt-4o-mini")

# csv 파일 로드.
loader_1 = CSVLoader('Cultural_asset.csv',encoding='UTF8')
loader_2 = CSVLoader('Travel_spot.csv',encoding='UTF8')

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
    ("system", "You are a helpful assistant. Please provide detailed, relevant recommendations based on the context provided. Ensure that all recommendations share the same city, district, and road name, and respond only with information from the data provided. Answer in Korean."),
    ("user", "Context: {context}\\n\\nQuestion: {question}. Provide clear and specific recommendations based on the data.")
])


# 디버깅을 위해 만든 클래스 (신경쓰지 않으셔도 됩니다)
class SimplePassThrough:
    def invoke(self, inputs, **kwargs):
        return inputs

# 프롬프트 클래스
class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        # response_docs 내용을 trim해줌 (가독성을 높여줌)
        if isinstance(inputs, list): # inputs가 list인 경우. 즉 여러개의 문서들이 검색되어 리스트로 전달된 경우
            context_text = "\n".join([doc.page_content for doc in inputs]) # \n을 구분자로 넣어서 한 문자열로 합쳐줌
        else:
            context_text = inputs # 리스트가 아닌경우는 그냥 리턴해줌

        # 프롬프트
        formatted_prompt = self.prompt_template.format_messages( # 템플릿의 변수에 삽입해줌
            context=context_text, # {context} 변수에 context_text, 즉 검색된 문서 내용을 삽입함
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
        # 1단계 : query를 리트리버에 넣어주고, response_docs를 얻어모
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

st.sidebar.title("지도")
with st.sidebar:
    add_radio = st.map(data,latitude='lat', longitude='lon')


# 날짜
# 오늘 날짜 자동 설정
today = datetime.date.today()

# 사이드바에 날짜 입력 위젯을 추가하고, 오늘 날짜를 기본값으로 설정
travel_date = st.sidebar.date_input("날짜를 클릭 하세요", today)

# 선택된 날짜 출력할껀가요? 일단 적어두겠습니다
# st.write(f"선택한 날짜: {travel_date}")

#기존 코드
st.sidebar.title("달력")
with st.sidebar:
    travel_date = st.date_input("날짜를 클릭 하세요", datetime.date(2024, 12, 1))
    
    
import time

# 로딩이 좀 걸리길래 넣어봤습니다 쓸모없으면 삭제!
# 진행률 표시
# 진행률 표시
progress_bar = st.progress(0)

# 100번의 반복을 통해 진행률 표시
# 100인 이유는 100%~
for i in range(100):
    time.sleep(0.1)
    progress_bar.progress(i + 1)


# 실시간 지역별 온도 업데이트(뉴스처럼)
#import requests
# 중요!!!! OpenWeatherMap API 가입하고 가져와야함 그래서 쓸껀지,말껀지 등 여쭤보려합니다
# API_KEY 위에서 부름
#def get_current_temperature(region):
#    url = f"http://api.openweathermap.org/data/2.5/weather?q={region},kr&appid={API_KEY}&units=metric"
#    response = requests.get(url)
#    data = response.json()
#    return data["main"]["temp"]  # 섭씨 온도를 반환

# 지역 리스트
#regions = ["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon", "Ulsan", "Gangneung", "Jeju"]

# 비어있는 컨테이너 생성
#metric_containers = {region: st.empty() for region in regions}

# 실시간 온도 업데이트
#for _ in range(100):
#    for region in regions:
#        current_temperature = get_current_temperature(region)
#        delta = random.choice([1, -1])  # 온도의 변화량
#        new_temperature = current_temperature + delta
#       
#        # 각 지역의 메트릭 업데이트
#        metric_containers[region].metric(
#            label=f"{region} 현재 온도",
#            value=f"{new_temperature}°C",
#            delta=f"{delta}°C"
#        )   
#    # 5초 대기
#    time.sleep(5)



# 진행률이 끝난 후 타이틀 표시, 타이틀 사용 시 이건 뭘 사용할 지 정해야할 것 같습니다~~~
st.title("77M PROJECT!")

# 공지글 표시 (투명한 배경과 주황색 글씨)
st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.5); padding: 10px; border-radius: 5px;">
        <strong style="color: orange;">▶</strong> 문화재 주변의 여행지 추천 챗봇입니다!<br>  
        대한민국의 궁금한 국보나 보물을 입력하면 주변 여행지를 추천해드립니다.<br>  
        " 000의 근방 추천해줘 "라고 입력해주세요!
    </div>
""", unsafe_allow_html=True)


if query := st.chat_input("대한민국의 국보 또는 보물을 입력하시거나 물어보세요"):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 디버깅: 쿼리 및 리트리버 출력 확인
    print(f"Query: {query}")
    
    # 1. 리트리버로 question에 대한 검색 결과를 response_docs에 저장함
    response_docs = rag_chain_debug["context"].invoke({"question": query})
    print(f"Retrieved documents: {response_docs}")

    # 2. 프롬프트에 질문과 response_docs를 넣어줌
    prompt_messages = rag_chain_debug["prompt"].invoke({
        "context": response_docs,
        "question": query
    })
     # 3. 완성된 프롬프트를 LLM에 넣어줌
    response = rag_chain_debug["llm"].invoke(prompt_messages)
    with st.chat_message("assistant"):
        if response and hasattr(response, 'content') and response.content:
            st.markdown(response.content)
        else:
            st.markdown("죄송합니다. 관련 정보를 찾을 수 없습니다.")


  
    
    
    
    
