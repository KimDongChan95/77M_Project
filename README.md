## 목차

1.  **프로젝트 개요**
2.  **팀원 구성**
3.  **프로젝트 일정**
4.  **기능 소개**
5.  **기술적 의사결정 및 주요기능 소개**
6.  **트러블 슈팅**
7.  **사용 기술**
8.  **성과 및 회고**
-----
#### **1. 프로젝트 개요** {#프로젝트-개요}

바쁜 사회 속에서, 국내문화와 관광에 대한 한국인들의 관심이 시들어가는 현상을 개선하고, 또한 발전하는 관광 산업을 통해 외국인 관광객들에게도 더 쉽고 편리한 접근을 하기 위해 최근 발전하고 있는 AI 챗봇 기술을 활용해서 문화재 주변 여행지의 검색 시스템을 기획하게 되었습니다. 사용자가 대한민국의 ‘문화재(국보나 보물)’에 대해 검색하면, 이를 바탕으로 관련된 다른 장소나 주변 명소를 추천받을 수 있어, 문화재뿐만 아니라 그 주변 명소까지 알 수 있는 보다 다양한 경험을 제공하려는 목적입니다.

-----
#### **2. TEAM 77M🤖**

| 팀장 | 김동찬 | 프로젝트 관리, 일정 조율, 회의 주재, 지라(JIRA) 관리. 챗봇 동작 검증 |
| --- | --- | --- |
| 데이터 크롤링 담당 | 박소연 | llm과 streamlit 합친 데이터에 대한 기능 추가 및 수정 |
| 추천 시스템 개발 | 한상민 | 여행지 추천 알고리즘 설계 및 구현, AI 모델 훈련 및 검증. 여행지 데이터 크롤링 및 CSV 저장 |
| 웹 인터페이스 개발 | 서승화 | Streamlit 기반 사용자 인터페이스 설계 및 구현. |
| 테스터 | 박수호 | 챗봇 동작 검증, 오류 및 사용성 테스트, 코드 디버깅 |
-----
#### **3. 프로젝트 일정** {#프로젝트-일정}

| 날짜 | 업무 |
| --- | --- |
| 11.20 ~ 11.24 | 프로젝트 기획 |
| 11.25 ~ 11.29 | 웹크롤링 및 데이터 전처리 |
| 11.30 ~ 12.01 | LLM 모델 구축, Streamlit UI 설정 |
| 12.02 | 성능 테스트 |
| 12.03 | 발표 준비 및 보고서 작성 |
| 12.04 | 프로젝트 발표 |
-----
#### **4. 기능 소개** {#기능-소개}

- 우리나라의 문화재(국보 및 보물)를 입력하면 근방의 명소를 추천합니다.

![와이어프레임 이미지](https://github.com/KimDongChan95/77M_Project/blob/hsm/%EC%99%80%EC%9D%B4%EC%96%B4%ED%94%84%EB%A0%88%EC%9E%84.jpg?raw=true)

- 채팅창에 문화재(국보 및 보물)를 입력하면 입력한 챗봇이 문서에서 관련 정보를 찾아 질문에 대한 답변을 생성합니다. 이로 인해 근처에 있는 명소의 정보(주소, 전화 번호, 이름)를 확인할 수 있습니다.
- 질문 중 키워드를 뽑아내 대화 기록에 키워드로 저장하고, 키워드를 클릭하여 이전에 대화했던 내용을 다시 한번 확인할 수 있습니다.
- 달력을 사용하여 오늘의 날짜를 확인할 수 있습니다.
- 지도를 사용하여 이동 및 확대, 축소를 활용하여 챗봇이 추천해준 명소의 위치를 확인할 수 있습니다.
- 상단에 로딩창을 표시하여 챗봇이 대화를 생성하는 데 얼마나 걸리는지 확인할 수 있습니다.
- 화면 중앙에 경복궁의 이미지를 삽입하여 멋진 문화재 근처에는 어떤 명소가 있을지 궁금증을 유발합니다.

주요 기능으로는 이 챗봇의 주 목적인, ‘문화재와 관련된 명소 검색 및 정보 제공’이 있습니다. 사용자는 특정 문화재(국보 혹은 보물)에 대해, 주어진 조건에 맞는 형식으로 이름을 입력하면, AI 챗봇은 해당 문화재와 관련해서 주변 명소의 ‘명칭’과 ‘주소’ 및 명소와 관련된 ‘간단 설명’을 추가해서 출력값으로 나타내는 기능을 가지고 있습니다.

-----
#### **5. 기술적 의사결정 및 주요기능 소개** {#기술적-의사결정-및-주요기능-소개}

AI 챗봇을 통해 사용자는 자연어로 질문을 하거나 명확한 추천 장소를 검색할 수 있어야 하므로, 자연어 처리(NLP) 성능이 중요한 요소입니다. 이를 위해 최신의 AI 챗봇 엔진을 선택해야 하기에 OPENAI의 GPT 모델을 선택하게 되었습니다. 또한, 검색 시스템은 문화재와 관련된 명소 정보 연결하는 역할을 해야 하므로, 외부 API 또는 데이터베이스와의 통합이 필요합니다.

<details>
<summary>더보기</summary>


#### 텍스트 유사도 비교 방식 선택

챗봇 프로젝트에서 텍스트 유사도 비교방식을 결정하기 위해 벡터기반 유사도비교와 FuzzyWuzzy 라이브러리를 비교하고, 각각의 장단점을 검토한 후 기술적 선택을 진행하였습니다. 

#### 벡터 기반 방식의 선택 이유

벡터 기반 방식을 최종 선택했는데 그 이유는 벡터 방식은 동의어와 문맥적 유사도를 효과적으로 인식하며, 다양한 언어와 긴 텍스트에서도 높은 정확도를 제공합니다. 데이터의 양이 많지 않다면 FuzzyWuzzy 라이브러리를 사용하는 것이 간단한 구현과 빠른 테스트에 유리하지만, 데이터의 양이 많거나 향후 복잡한 기능들이 추가될 것을 고려하여 벡터기반 유사도비교가 사용자의 문맥적 의도를 더 정확히 파악할 수 있기 때문에 더 적합하다고 판단했습니다.

#### 텍스트 분할 방법

텍스트가 특정 구분자로만 나누어도 충분하고, 구분자 하나로도 텍스트를 적절히 나눌 수 있기 때문에 CharacterTextSplitter를 사용했습니다.

#### OpenAiEmbeddings의 선택

OpenAiEmbeddings은 강력하게 언어모델과 상호작용하여 텍스트 임베딩을 생성해주고 단어나 문장의 의미를 포착하며, 기계학습모델이 텍스트 데이터를 이해하고 처리할 수 있도록 하기 때문에 선택되었습니다.

#### 결과물 표시

개발한 LLM모델의 결과물을 보여주기 위해 A faster way to build and share data apps인 Streamlit을 사용하였습니다.


</details>

주요 기능으로는 이 챗봇의 주 목적인, ‘문화재와 관련된 명소 검색 및 정보 제공’이 있습니다. 사용자는 특정 문화재(국보 혹은 보물)에 대해, 주어진 조건에 맞는 형식으로 이름을 입력하면, AI 챗봇은 해당 문화재와 관련해서 주변 명소의 ‘명칭’과 ‘주소’ 및 명소와 관련된 ‘간단 설명’을 추가해서 출력값으로 나타내는 기능을 가지고 있습니다.

-----
#### **6. 트러블 슈팅** {#트러블-슈팅}

<details>
<summary>크롤링 과정의 주요 이슈 및 해결 방안</summary>

### 문제 원인
1. **문화재 위키백과 사이트 개편**: 갑작스러운 사이트 개편으로 인해 기존 크롤링 방식이 무효화되었습니다.
2. **여행지 사이트의 복잡성**: 팀원들이 학습한 지식만으로는 여행지 사이트의 크롤링이 어려웠습니다.
3. **리스트 변경**: 보물이 국보로 승격되거나, 화재 및 소실로 인해 문화재 지정이 해제되는 등의 이유로 리스트가 변경되었습니다.
4. **동적 크롤링의 한계**: 
   - 김동찬님이 Selenium을 활용한 동적 크롤링을 시도했으나, 작업 시간이 2~3시간 소요되었습니다.
   - 실패 시 다음 페이지로 넘어가지 못해 크롤링이 중단되는 문제가 있었습니다.
   - 3994개의 페이지 중 2600페이지에서 작업이 중단되는 한계를 겪었습니다.

### 해결 방법
1. **HTML 구조 분석**: 한상민님이 HTML 구조를 분석하여 여행지 사이트의 크롤링 문제를 해결하였습니다.
2. **코드 수정**: 실패한 페이지를 다시 크롤링하거나 생략하고 넘어가는 방식으로 코드를 수정했으나, 여전히 한계가 있었습니다.
3. **정적 크롤링으로의 전환**: 동적 크롤링의 한계를 깨달은 후, 해당 사이트에서 고정된 HTML 구조와 효율적인 데이터 수집을 통해 정적 크롤링이 더 안정적이고 효율적이라는 것을 인식했습니다.

</details>

<details>
<summary>데이터 전처리 과정의 주요 사항</summary>



- **문제 인식**: 보물에서 해제된 데이터는 지정일 열에 "승격", "해제", "소실", "재지정"이라는 텍스트가 포함된다는 공통점이 있었습니다.
  
- **전처리 방법**:
  - 팀원 모두가 파이썬 판다스(Pandas)를 활용하여 다음 코드를 사용했습니다:
    ```python
    df = df[~df['지정일'].str.contains('승격|해제|소실|재지정', na=False)]
    ```
    - 이 코드는 해당 문구가 포함된 행을 제거하는 역할을 했습니다.
  - 또한, 중복된 행을 제거하기 위해 다음 코드를 사용했습니다:
    ```python
    df.drop_duplicates()
    ```

- **문제 발생**: 크롤링 코드와 이를 융합하는 과정에서 어려움을 겪었고, 시간상의 제한으로 인해 취소선이 표시된 국보나 보물을 일일이 확인하는 방식으로 작업을 진행해야 했습니다.

- **작업 분담**: 해당 작업은 소연님께서 맡아주시고, 최종적으로 57개의 취소선 데이터를 확인해주셨습니다.

</details>

<details>
<summary>LLM 모델링 과정의 주요 사항</summary>

### 모델 구현
- **김동찬님**은 문자열 간의 유사도를 비교하고 텍스트 매칭을 수행하는 **fuzzywuzzy** 라이브러리를 활용하여:
  - 로딩 속도가 빠르고 (거의 실시간으로 출력됨)
  - 입력에 대한 응답 정확도가 높은 (크롤링한 데이터에서 데이터 가져올 확률 90%) 모델을 구현했습니다.

### 팀 프로젝트 방향성
- 그러나 이번 팀 프로젝트의 취지에 비추어보면, **RAG**와 **RAGchain** 기술을 사용해 배운 내용을 활용하는 것이 더 적합하다는 **박수호님**의 의견이 있었습니다.
- 이에 따라, **FAISS**를 Retriever로 변환하고 RAGchain 기술을 배운 것을 토대로 **한상민님**이 구현하신 LLM 모델링을 저희 챗봇의 기초로 채택하여 프로젝트를 진행하게 되었습니다.

### 리소스 문제
- 많은 양의 페이지를 로드하다 보니 시간 지연 및 메모리 부족 등 리소스 문제가 발생했습니다.
- 이 문제를 해결하기 위해, 추후 **FAISS DB**를 로컬로 저장하는 방안으로 로딩에 소요되는 시간 및 메모리 면에서 더 효율적인 작업을 할 수 있었습니다.

</details>

<details>
<summary>Streamlit UI 제작 과정의 주요 사항</summary>

- **오류 상황**: Streamlit UI 제작 과정 중 버전에 따라 오류가 발생하는 경우가 있었지만, 그 외에는 큰 문제는 없었습니다.
  
- **LLM 모델링과 Streamlit 통합**: 
  - LLM 모델링 시간의 지연이 Streamlit UI에 영향을 미쳤습니다.
  - 이를 보다 단축하기 위해 여러 번의 코드 수정이 진행되었습니다.
    - 예를 들어, `time.sleep()`을 0.01초로 설정하는 등의 방법이 사용되었습니다.
   
</details>

-----

#### **7\. 사용 기술** {#사용-기술}

🖥️ 프론트엔드

-   Streamlit

📀 백엔드

-   Python
-   Beautiful Soup
-   인공지능
    -   OpenAI
    -   LangChain

💬 협업도구

-   GitHub
-   Slack
-   jira
-   Notion
-----
#### **8\. 성과 및 회고** {#성과-및-회고}

-   잘된 점
    -   필요한 정보를 웹 크롤링을 통해 잘 수집 하였고 임배딩하여 RAG에 적합한 자료로 변환하였다.
    -   협업 도구들로 파일을 공유하고, 문제가 생긴 경우 모두가 해결하기위해 노력함.
    -   Vector DB학습으로 인해 streamlit 실행이 매우 오래 걸렸으나, Vector DB를 저장후 불러오기를 사용함으로써 streamlit실행속도와 답변출력속도가 매우 빨라지도록 개선됨.
-   아쉬운 점
    -   첫 프로젝트로 계획을 짜는데 여러 에러사항들로 인해, 마감일이 다가올수록 시간에 쫓김
    -   github 사용에 대한 미숙으로 협업도구는 slack을 주로 사용함.
    -   streamlit오류를 해결하는 과정에서, 가상환경 재설치를 여러번하여 시간소요가 많았음.
    -   입력에 대한 출력을 번역하여 여러나라의 언어(중국어, 일본어, 영어)로 표현하고 싶었으나, ai모델을 2개이상 불러오기를 할 수 없다는 한계를 어떻게 극복해야할지 방법을 찾지못함.
    -   입력과 출력을 세션에 저장하여 사이드바에서 표기하고 열어볼 수 있지만, 입력을 자유롭게 하지못하고 특정문장 형태로만 입력해야 오류가 발생하지않음.
    -   Vector DB에 데이터프레임형태로 데이터를 넣고 싶었으나 실패함.
    -   문화재 이름이 길거나, 문화재를 구성하는 단어간 띄어쓰기를 잘 인식하지 못함.
-    향후 계획
    -   한국어만 아닌, 다국어 지원 기능 구현
    -   문화재 이름에 포함되는 띄어쓰기 및 지역명이 들어간 문화재를 명확히 인식하는 기능 구현
    -   다양한 질문의 입력을 받을 수 있도록 구현
    -   챗봇의 답변에 관련해서 텍스트를 음성으로 입ㆍ출력기능 구현
    -   챗봇의 답변으로 검색되는 명소 시설 및 명소 지역의 현재 날씨 알림 기능 구현
    -   위도 경도를 이용한 상세 거리표시 기능 구현!([Geocoding](https://ko.wikipedia.org/wiki/%EC%A7%80%EC%98%A4%EC%BD%94%EB%94%A9))
