import streamlit as st
import torch
import os
import pdfplumber

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BertForSequenceClassification
)

# =====================================
# PATHS (YOUR FOLDERS)
# =====================================

BASE_DIR = r"D:\Legal_AI_App"

CLASSIFIER_PATH = os.path.join(BASE_DIR, "bert_classifier_model")
SUMMARIZER_PATH = os.path.join(BASE_DIR, "t5_summarizer_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# LOAD MODELS
# =====================================

@st.cache_resource
def load_models():

    # ---- BERT CLASSIFIER ----
    tokenizer_cls = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_cls = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model_cls.to(device)
    model_cls.eval()

    # ---- T5 SUMMARIZER ----
    tokenizer_sum = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model_sum = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model_sum.to(device)
    model_sum.eval()

    return tokenizer_cls, model_cls, tokenizer_sum, model_sum


# =====================================
# PDF TEXT EXTRACTION
# =====================================

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()


# =====================================
# CATEGORY PREDICTION
# =====================================

LABELS = ["Civil", "Constitutional", "Criminal", "Service", "Tax", "Other"]

def predict_category(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = outputs.logits.argmax().item()

    if pred_id < len(LABELS):
        return LABELS[pred_id]
    return "Other"


# =====================================
# CLEAN SUMMARY TEXT
# =====================================

def clean_summary(text):

    junk_words = [
        "Indian Kanoon",
        "http://",
        "www.",
        "AIR",
        "SCC"
    ]

    for word in junk_words:
        text = text.replace(word, "")

    return text.strip()


# =====================================
# SUMMARIZATION (IMPROVED QUALITY)
# =====================================

def summarize_text(text, tokenizer, model):

    prompt = (
        "Summarize the following legal case in simple everyday language as if explaining to a normal person. Avoid legal jargon. Explain facts, issue, and decision clearly: "
        "Include background, legal issue, arguments, and final court decision in detail:\n\n"
        + text[:3500]   # increased input length
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768   # increased token space
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],

            max_length=500,        # 🔼 longer summary
            min_length=300,        # 🔼 avoid tiny summaries

            num_beams=6,
            repetition_penalty=2.5,
            length_penalty=1.0,
            no_repeat_ngram_size=4,

            temperature=0.7,       # makes language smoother
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return clean_summary(summary)


# =====================================
# STREAMLIT UI
# =====================================

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
        text = extract_text_from_pdf(uploaded_file)

    if len(text) < 200:
        st.error("Not enough text extracted from PDF.")
    else:

        with st.spinner("Analyzing document..."):

            category = predict_category(
                text,
                tokenizer_cls,
                model_cls
            )

            summary = summarize_text(
                text,
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
