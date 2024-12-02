import os
import streamlit as st
import pandas as pd
import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# 환경 변수 로드
load_dotenv()

# OpenAI API key 설정
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model="gpt-4o-mini")

# 문서 로드 및 처리
loader1 = CSVLoader("sum_sum_Cultural2.csv", encoding='UTF8')
loader2 = CSVLoader("place.csv", encoding='UTF8')

Treasures = loader1.load()
Travel = loader2.load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_overlap=10, length_function=len)
splits = text_splitter.split_documents(Treasures)
splits2 = text_splitter.split_documents(Travel)


# 두 개의 splits 리스트를 합치기(청크 더이상xx)
combined_splits = splits + splits2

# 벡터화 및 스토어 생성
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(documents=combined_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# 프롬프트 템플릿 설정
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

# LLMChain 설정
llm_chain = LLMChain(llm=model, prompt=contextual_prompt)

# Streamlit UI 설정
st.title("77M🤖")

# 사용자 입력
prompt = st.chat_input("대한민국의 국보 또는 보물을 입력하시거나 물어보세요")

# 사용자 질문에 대해 OpenAI 응답 생성
def openai_stream(question):
    try:
        # 질문에 대한 응답 생성
        response_Treasures = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in response_Treasures])
        
        # 프롬프트 템플릿에 데이터 적용
        prompt_messages = contextual_prompt.format_messages(
            context=context_text,
            question=question
        )
        
        # LLM으로 응답 생성
        response = llm_chain.invoke(prompt_messages)
        return response.content
    except Exception as e:
        st.error(f"Error: {str(e)}")

# 사용자 질문 처리
if prompt:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})    

    with st.chat_message("assistant"):
        response = openai_stream(prompt)
        st.markdown(response)  # 응답 출력
    st.session_state.messages.append({"role": "assistant", "content": response})

# 사이드바 설정: 지도 및 날짜
data = pd.DataFrame({'lat': [37.56], 'lon': [127]})
st.sidebar.title("지도")
st.sidebar.map(data, latitude='lat', longitude='lon')

st.sidebar.title("달력")
travel_date = st.sidebar.date_input("날짜를 클릭 하세요", datetime.date(2024, 12, 1))
