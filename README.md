# AI-Legal-Assistant
📚 GPT-4와 LangGraph 기반의 AI 법률 상담 서비스. 사용자의 질문을 분석하고, Pinecone에 벡터화된 법률 PDF 문서를 검색하여 정확하고 근거 기반의 법률 상담을 제공합니다. Streamlit UI와 PDF 리포트 출력 기능까지 지원합니다.

# 🧑‍⚖️ AI 법률 상담 시스템 (LangGraph + Streamlit)

GPT-4와 LangGraph, Pinecone, Streamlit을 활용한 **AI 기반 법률 상담 서비스**입니다.  
사용자의 질문을 분석하고, 관련 법령을 기반으로 판단과 요약을 제공하며, 상담 결과를 PDF로 저장할 수 있습니다.

---

## ✅ 주요 기능

### 1. 질문 분석 (HITL 포함)
- 질문 내용을 GPT-4가 분석하여 법률 분야 및 쟁점을 도출
- 사용자가 직접 분석 결과를 수정 가능 (Human-in-the-Loop)

### 2. 벡터 검색 기반 문서 검색
- OpenAI 임베딩 모델로 벡터화한 법령 PDF들을 Pinecone에 저장
- 질문에 맞는 관련 조항을 벡터 기반으로 검색

### 3. 근거 기반 판단 생성 (HITL 포함)
- 검색된 문서를 바탕으로 GPT-4가 판단 및 해석 생성
- 사용자 검토 및 수정 가능

### 4. 요약 및 조치 제안
- 판단 결과를 이해하기 쉽게 요약
- 현실적인 조치 방법 제안

### 5. PDF 리포트 생성
- 상담 내용을 정리한 **법률 상담 보고서** 자동 생성
- NanumGothic 폰트를 활용한 한글 문서 지원

---

## 🖥️ 사용 기술

- [x] **LangGraph** - Agent 기반 흐름 제어
- [x] **LangChain** - 문서 로딩 및 임베딩
- [x] **Pinecone** - 벡터 DB
- [x] **OpenAI GPT-4 / Embeddings** - 질문 이해, 임베딩
- [x] **Streamlit** - 웹 UI 구현
- [x] **FPDF** - PDF 보고서 생성
