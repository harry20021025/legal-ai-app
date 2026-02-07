import streamlit as st
import torch
import pdfplumber

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
            if page.extract_text():
                text += page.extract_text() + " "
    return text.strip()


# ==============================
# CASE CLASSIFICATION
# ==============================

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


# ==============================
# CLEAN SUMMARY
# ==============================

def clean_summary(text):

    junk = ["Indian Kanoon", "http://", "www.", "AIR", "SCC"]

    for j in junk:
        text = text.replace(j, "")

    return text.strip()


# ==============================
# BULLET STYLE SUMMARY
# ==============================

def summarize_text(text, tokenizer, model):

    prompt = f"""
You are a legal assistant.

Summarize the following court judgment in SIMPLE language.
Do NOT copy text.
Do NOT repeat words.
Use BULLET POINTS only.

Format exactly like this:

• Background:
• Main Legal Issue:
• Key Arguments:
• Court’s Final Decision:

Judgment text:
{text[:3500]}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],

            max_length=450,
            min_length=220,

            num_beams=8,
            no_repeat_ngram_size=4,
            repetition_penalty=2.8,

            top_p=0.9,
            do_sample=False,

            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_summary(summary)



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
