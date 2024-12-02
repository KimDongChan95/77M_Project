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
loader_1 = CSVLoader('final.csv',encoding='UTF8')
# loader_2 = CSVLoader('place.csv',encoding='UTF8')

Tresures = loader_1.load()
# Travel = loader_2.load()

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(Tresures)
# splits += recursive_text_splitter.split_documents(Travel)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})


# 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])

# 디버깅을 위해 만든 클래스 (신경쓰지 않으셔도 됩니다.)
# class SimplePassThrough:
#     def invoke(self, inputs, **kwargs):
#         return inputs

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
st.sidebar.title("달력")
with st.sidebar:
    travel_date = st.date_input("날짜를 클릭 하세요", datetime.date(2024, 12, 1))
    

if query := st.chat_input("대한민국의 국보 또는 보물을 입력하시거나 물어보세요"):
    # 입력 메세지 저장
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 화면에 입력한 메세지 출력(Markdown 형식)
    with st.chat_message("user"):
        st.markdown(query)
    # 사용자가 입력한 메세지를 session_state.messages에 저장
    st.session_state.messages.append({"role": "user", "content": query}) 
    # 0. 질문을 받아서 query에 저장함
    # query = input("질문을 입력하세요 : ")


    # 1. 리트리버로 question에 대한 검색 결과를 response_docs에 저장함
response_docs = rag_chain_debug["context"].invoke({"question": query})

    # 2. 프롬프트에 질문과 response_docs를 넣어줌
prompt_messages = rag_chain_debug["prompt"].invoke({
    "context": response_docs,
    "question": query
})

    # 3. 완성된 프롬프트를 LLM에 넣어줌
response = rag_chain_debug["llm"].invoke(prompt_messages)
stream=True  # 실시간으로 입력 송출(False면 입력이 완료된 후에 송출)

    # print("\n답변:")
    # print(response.content)'
    
answer_container = st.empty()

full_response = ""

    # 응답을 스트리밍으로 하나씩 출력
# for chunk in response:
#     if chunk != None:
#         full_response += chunk
#         answer_container.markdown(full_response)  # 실시간으로 출력
#     st.write(full_response)
    # openai의 응답
with st.chat_message("assistant"):
    st.markdown(response.content)
        # response = openai_stream(prompt) 
    # openai의 응답을 session_state.messages에 저장
    # st.session_state.messages.append({"role": "assistant", "content": response.content})
    
    
    # TypeError: argument 'text': 'NoneType' object cannot be converted to 'PyString' 이게뭔지 모르겠음......슈발?