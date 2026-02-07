import streamlit as st
import torch
import pdfplumber
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BertForSequenceClassification
)

# ==============================
# HUGGINGFACE ONLINE MODELS
# ==============================

CLASSIFIER_PATH = "hari102002/legal-bert-classifier"
SUMMARIZER_PATH = "hari102002/legal-t5-summarizer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODELS
# ==============================

@st.cache_resource
def load_models():
    tokenizer_cls = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
    model_cls = BertForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
    model_cls.to(device).eval()

    tokenizer_sum = AutoTokenizer.from_pretrained(SUMMARIZER_PATH)
    model_sum = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_PATH)
    model_sum.to(device).eval()

    return tokenizer_cls, model_cls, tokenizer_sum, model_sum

# ==============================
# PDF TEXT EXTRACTION
# ==============================

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

# ==============================
# LEGAL TEXT PREPROCESSING
# ==============================

def preprocess_legal_text(text):
    # Remove page numbers, exhibits, noisy brackets
    text = re.sub(r"\b\d+\s+of\s+\d+\b", "", text)
    text = re.sub(r"\(\(.*?\)\)", "", text)
    text = re.sub(r"Exh\.\s*\d+", "", text)
    text = re.sub(r"\s+", " ", text)

    # Remove citations clutter
    text = re.sub(r"\(\d{4}\).*?\)", "", text)

    return text.strip()

def remove_front_matter(text):
    for key in ["JUDGMENT", "Judgment", "REASONS", "Reasons"]:
        idx = text.find(key)
        if idx != -1:
            return text[idx:]
    return text

# ==============================
# TEXT CHUNKING
# ==============================

def chunk_text(text, max_words=700):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]

# ==============================
# CHUNK SUMMARIZATION
# ==============================

def summarize_chunk(chunk, tokenizer, model):
    prompt = (
        "Summarize this portion of a legal judgment clearly.\n"
        "Focus only on facts, legal issue, reasoning, and outcome.\n\n"
        + chunk
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=200,
            min_length=110,
            num_beams=5,
            repetition_penalty=2.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ==============================
# FINAL SUMMARY CLEANUP
# ==============================

def clean_summary(text):
    junk = ["Indian Kanoon", "http://", "www.", "AIR", "SCC"]
    for j in junk:
        text = text.replace(j, "")
    return text.strip()

# ==============================
# FULL CHUNKED SUMMARIZATION
# ==============================

def summarize_text(text, tokenizer, model):

    # 🔹 Preprocess text
    text = preprocess_legal_text(text)
    text = remove_front_matter(text)

    # 🔹 Chunking
    chunks = chunk_text(text)
    chunk_summaries = []

    for chunk in chunks:
        chunk_summaries.append(
            summarize_chunk(chunk, tokenizer, model)
        )

    combined_summary = " ".join(chunk_summaries)

    # 🔹 Final refinement (MOST IMPORTANT)
    final_prompt = (
        "You are a legal assistant.\n\n"
        "Using the summaries below, write a clean bullet-point summary "
        "for a common person.\n\n"
        "Use EXACTLY this structure:\n"
        "• Background\n"
        "• Main Legal Issue\n"
        "• Court’s Reasoning\n"
        "• Final Court Decision\n\n"
        "Do NOT include party addresses, exhibit numbers, citations, or procedural clutter.\n\n"
        + combined_summary
    )

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        final_output = model.generate(
            inputs["input_ids"],
            max_length=450,
            min_length=260,
            num_beams=6,
            repetition_penalty=2.6,
            no_repeat_ngram_size=4,
            early_stopping=True
        )

    final_summary = tokenizer.decode(
        final_output[0],
        skip_special_tokens=True
    )

    return clean_summary(final_summary)

# ==============================
# CASE CLASSIFICATION
# ==============================

LABELS = ["Civil", "Constitutional", "Criminal", "Service", "Tax", "Other"]

def predict_category(text, tokenizer, model):
    text = preprocess_legal_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return LABELS[outputs.logits.argmax().item()]

# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Legal AI System", layout="centered")

st.title("⚖️ Legal Document AI System")

st.write("""
Upload a court judgment PDF and get:

• 📂 Case category  
• 📄 Simple human-readable summary  
""")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Loading AI models..."):
        tokenizer_cls, model_cls, tokenizer_sum, model_sum = load_models()

    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    if len(raw_text) < 200:
        st.error("Not enough text extracted from PDF.")
    else:
        with st.spinner("Analyzing document..."):
            category = predict_category(
                raw_text,
                tokenizer_cls,
                model_cls
            )

            summary = summarize_text(
                raw_text,
                tokenizer_sum,
                model_sum
            )

        st.success("✅ Analysis complete")

        st.subheader("📂 Predicted Case Type")
        st.write(category)

        st.subheader("📄 Simplified Summary")
        st.write(summary)

st.markdown("---")
st.caption("Built by Hariom Dixit | Legal NLP AI System")
