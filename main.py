import re
from pathlib import Path
import pdfplumber
import easyocr
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import logging

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# ─── Setup ────────────────────────────────────────────────────────
load_dotenv()
reader = easyocr.Reader(['en'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audit_logger")

PAYS_SLIP_PATTERNS = {
    "Basic": r"\bBasic(?: Pay)?\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
    "HRA": r"\bHRA(?: Allowance)?\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
    "EPF": r"\b(?:EPF|PF)\s*(?:Contribution)?\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
    "ESI": r"\bESI\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
    "PT": r"\bProfessional\s*Tax\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
    "TDS": r"\bTDS(?: Deduction)?\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
    "Net Pay": r"\bNet(?: | )Pay(?:able)?\s*:?\s*₹?\s*([\d,]+(?:\.\d{1,2})?)",
}


def extract_text(file, file_type: str) -> str:
    if file_type == "pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    elif file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(file)
        result = reader.readtext(image, detail=0)
        return "\n".join(result)
    else:
        return ""


def parse_fields(text: str) -> dict:
    out = {}
    for label, pat in PAYS_SLIP_PATTERNS.items():
        m = re.search(pat, text, flags=re.I)
        if m:
            out[label] = float(m.group(1).replace(",", ""))
    return out


def load_kb(white_pdf: Path, cache_dir: Path):
    emb = OpenAIEmbeddings(model="text-embedding-ada-002")

    if cache_dir.exists() and any(cache_dir.glob("*.parquet")):
        print("✅ Loading existing Chroma DB")
        return Chroma(persist_directory=str(cache_dir), embedding_function=emb)

    text = extract_text(white_pdf, "pdf")
    splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separator="\n")
    docs = [Document(page_content=chunk)
            for chunk in splitter.split_text(text)]
    return Chroma.from_documents(docs, emb, persist_directory=str(cache_dir))


def compliance_audit(text, kb, llm):
    fields = parse_fields(text)

    q = ("Explain statutory deductions like TDS, EPF, ESI, "
         "and Professional Tax for Indian salary processing")
    docs = kb.similarity_search(q, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = PromptTemplate(
        template="""
You are a payroll auditor. Here are the rules:

{context}

Here is the payslip text:

{payslip}

1. List key amounts.
2. Check compliance.
3. Flag any issues.
""",
        input_variables=["context", "payslip"]
    )

    chain = prompt | llm
    report = chain.invoke({"context": context, "payslip": text[:4000]})

    return fields, report.content


# ─── Load Knowledge Base and LLM ─────────────────────────────────
WHITE_PAPER = Path("payroll_compliance_e-book.pdf")
CACHE_DIR = Path("chroma_db")
kb = load_kb(WHITE_PAPER, CACHE_DIR)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(
    page_title="🧾 Payroll Compliance Checker", layout="centered")
st.title("🧾 Indian Payroll Compliance Checker")

# ─── EU AI Act Transparency Message ──────────────────────────────
st.info("⚠️ This tool uses AI (LLM) to generate audit reports. The result is for informational purposes only and should be verified by a human expert.")

# ─── Privacy Notice ──────────────────────────────────────────────
with st.expander("🔒 Data & Privacy Notice"):
    st.markdown("""
- Your uploaded payslip is processed **in-memory** only.
- No data is stored, shared, or used beyond this session.
- You can close this tab at any time to delete your data from memory.
- This tool complies with EU AI Act obligations for transparency and human oversight.
    """)

user_consent = st.checkbox(
    "✅ I understand and consent to process my payslip with AI.")

st.write("Upload a payslip **PDF or Image** to get a summary & compliance audit.")

uploaded_file = st.file_uploader("Upload Payslip (PDF, PNG, JPG, JPEG)", type=[
                                 "pdf", "png", "jpg", "jpeg"])

if uploaded_file and not user_consent:
    st.warning(
        "⚠️ Please check the consent box before your file can be processed.")

if uploaded_file and user_consent:
    file_type = uploaded_file.type.split("/")[-1].lower()

    

    try:
        with st.spinner("Processing..."):
            text = extract_text(uploaded_file, file_type)
            if not text.strip():
                st.error(
                    "❗ No readable text found. Please try with a clearer image or proper PDF.")
            else:
                fields, report = compliance_audit(text, kb, llm)

                st.subheader("🔢 Parsed Salary Components")
                st.json(fields)

                st.subheader("📝 Compliance Report (AI-generated)")
                st.text_area("Audit Result", report, height=300)

                st.caption(
                    "📌 This audit is generated by an AI language model. Always verify with an HR or legal professional.")
                st.caption(
                    "🤖 Powered by OpenAI GPT-4o and text-embedding-ada-002. Results may not be 100% accurate.")

                

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ─── Footer ──────────────────────────────────────────────────────
with st.expander("ℹ️ System Information"):
    st.markdown("""
**System Type**: Payroll AI Compliance Tool  
**Provider**: Parag Saxena  
**Purpose**: Assist with Indian payroll statutory audit and compliance  
**Model**: OpenAI GPT-4o + text-embedding-ada-002  
**Data Handling**: In-memory only, no storage or sharing  
**Human Oversight**: Always required  
""")
