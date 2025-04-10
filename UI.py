# LangGraph + Streamlit 기반 법률 상담 AI + PDF 출력 (한 파일 구성 + HITL 반영)

import os
from dotenv import load_dotenv
from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END
from pinecone import Pinecone
from fpdf import FPDF
import streamlit as st

# 1. 환경변수 및 모델 로딩
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "legal-docs-index"

# 2. Pinecone 인덱스 객체 가져오기
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# 3. 임베딩 및 벡터스토어 구성
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model
)

llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# 4. 상태 모델 정의
class AgentState(BaseModel):
    query: str
    context: Dict[str, Any] = {}

# 5. 에이전트 정의
analyze_prompt = SystemMessage(content="""
당신은 법률 상담 시스템의 질문 분석가입니다.
사용자의 질문에서 법률 분야(예: 민사, 형사, 노동)와 핵심 쟁점을 도출하세요.
""")

def analyze_agent(state: AgentState) -> AgentState:
    result = llm.invoke([analyze_prompt, {"role": "user", "content": state.query}])
    state.context["analysis"] = getattr(result, "content", str(result))
    return state

def retrieve_agent(state: AgentState) -> AgentState:
    docs = vectorstore.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(state.query)
    state.context["documents"] = docs
    return state

reason_prompt = SystemMessage(content="""
당신은 법률 전문가입니다. 아래 문서 내용을 바탕으로 질문에 대해 논리적으로 판단하고, 반드시 근거가 포함된 응답을 생성하세요.
""")

def reasoning_agent(state: AgentState) -> AgentState:
    doc_text = "\n\n".join([d.page_content for d in state.context.get("documents", [])])
    result = llm.invoke([reason_prompt, {"role": "user", "content": f"질문: {state.query}\n\n문서 내용:\n{doc_text}"}])
    state.context["answer"] = getattr(result, "content", str(result))
    return state

summarize_prompt = SystemMessage(content="""
당신은 법률 상담 요약가입니다. 아래 답변을 일반인이 이해하기 쉽도록 정리하고, 적절한 조치 방법을 제안하세요.
""")

def summarize_agent(state: AgentState) -> AgentState:
    result = llm.invoke([summarize_prompt, {"role": "user", "content": state.context.get("answer", "")}])
    state.context["summary"] = getattr(result, "content", str(result))
    return state

# 6. LangGraph 워크플로우 구성
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_agent)
workflow.add_node("retrieve", retrieve_agent)
workflow.add_node("reason", reasoning_agent)
workflow.add_node("summarize", summarize_agent)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "summarize")
workflow.add_edge("summarize", END)
app = workflow.compile()

# 7. Streamlit UI
st.set_page_config(page_title="AI 법률 상담")
st.title("📚 AI 법률 상담 서비스")

query = st.text_area("상담 내용을 입력하세요:")
run = st.button("상담하기")
save_pdf = st.button("📄 PDF 저장")

if run and query:
    with st.spinner("AI 상담 중입니다..."):
        state = AgentState(query=query)
        result = app.invoke(state)
        ctx = result.get("context", {})

        st.session_state["ctx"] = ctx
        st.session_state["query"] = query

        st.subheader("🔍 질문 분석")
        analysis_input = st.text_area("LLM 분석 결과를 검토 또는 수정하세요:", value=ctx.get("analysis", ""))
        ctx["analysis"] = analysis_input

        st.subheader("📘 판단 결과")
        answer_input = st.text_area("LLM 판단 결과를 검토 또는 수정하세요:", value=ctx.get("answer", ""))
        ctx["answer"] = answer_input

        st.subheader("✅ 요약 및 조치 제안")
        st.write(ctx.get("summary", "-"))

if save_pdf and "ctx" in st.session_state:
    class PDFReport(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)
            self.add_font('Nanum', '', 'NanumGothic.ttf', uni=True)
            self.add_font('Nanum', 'B', 'NanumGothic-Bold.ttf', uni=True)
            self.set_font('Nanum', '', 12)
            self.add_page()

        def header(self):
            self.set_font('Nanum', '', 14)
            self.cell(0, 10, '법률 상담 보고서', ln=True, align='C')
            self.set_font('Nanum', '', 10)
            self.cell(0, 10, f"상담일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='R')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Nanum', '', 9)
            self.cell(0, 10, f'- {self.page_no()} -', align='C')

        def section(self, title, content):
            self.set_font('Nanum', 'B', 12)
            self.cell(0, 10, title, ln=True)
            self.set_font('Nanum', '', 11)
            self.multi_cell(0, 8, content)
            self.ln()

    ctx = st.session_state["ctx"]
    query = st.session_state["query"]

    pdf = PDFReport()
    pdf.section("1. 사용자 질문", query)
    pdf.section("2. 질문 분석 결과", ctx.get("analysis", "-"))
    pdf.section("3. 판단 내용", ctx.get("answer", "-"))
    pdf.section("4. 요약 및 조치 제안", ctx.get("summary", "-"))

    output_path = f"법률_상담_보고서_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_path)

    st.success(f"PDF 저장 완료: {output_path}")
    with open(output_path, "rb") as f:
        st.download_button(label="📥 PDF 다운로드", data=f, file_name=output_path, mime="application/pdf")
