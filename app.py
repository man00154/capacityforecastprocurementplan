# app.py
import os
import json
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from datetime import timedelta

# -------- Config: use the user's model and endpoint ----------
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
# ------------------------------------------------------------

st.set_page_config(page_title="TinyDCPlanner — capacity forecast + procurement", layout="wide")

st.title("MANISH - DCPlanner — Forecast capacity & generate procurement plans")
st.markdown(
    """
pipeline:
- small lag-based ML forecast (RandomForest)
- tiny RAG (TF-IDF top-k retrieval)
- simple agentic 
"""
)

# ---------------- Utility functions ----------------

def read_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return df

def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df

def create_lag_features(series: pd.Series, lags: int = 7) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def train_forecaster(y: pd.Series, lags: int = 7) -> Dict[str, Any]:
    df_lag = create_lag_features(y, lags=lags)
    X = df_lag.drop(columns=["y"]).values
    y_tr = df_lag["y"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y_tr, test_size=0.2, random_state=42, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)
    last_window = y[-lags:].tolist()
    return {"model": model, "scaler": scaler, "last_window": last_window, "lags": lags}

def forecast_iterative(forecaster: Dict[str, Any], steps: int = 7) -> List[float]:
    model = forecaster["model"]
    scaler = forecaster["scaler"]
    last = forecaster["last_window"][:]
    lags = forecaster["lags"]
    preds = []
    for _ in range(steps):
        X = np.array(last[-lags:]).reshape(1, -1)
        Xs = scaler.transform(X)
        p = model.predict(Xs)[0]
        preds.append(float(p))
        last.append(p)
    return preds

# ---------------- Very simple RAG (TF-IDF) ----------------

class TinyRAG:
    def __init__(self):
        self.docs: List[str] = []
        self.tfidf = None
        self.vectors = None

    def index_texts(self, texts: List[str]):
        self.docs = texts
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
        self.vectors = self.tfidf.fit_transform(self.docs)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self.vectors or not self.docs:
            return []
        qv = self.tfidf.transform([query])
        sims = cosine_similarity(qv, self.vectors)[0]
        idxs = sims.argsort()[::-1][:top_k]
        results = [self.docs[i] for i in idxs if sims[i] > 0]
        return results

# ---------------- Minimal "agentic" orchestration ----------------

def forecast_tool(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> Dict[str, Any]:
    y = df[value_col].astype(float).reset_index(drop=True)
    lags = min(14, max(3, int(len(y) * 0.1)))
    forecaster = train_forecaster(y, lags=lags)
    preds = forecast_iterative(forecaster, steps=periods)
    last_date = pd.to_datetime(df[date_col].iloc[-1])
    try:
        delta = pd.to_datetime(df[date_col].iloc[-1]) - pd.to_datetime(df[date_col].iloc[-2])
    except Exception:
        delta = timedelta(days=1)
    future_dates = [last_date + (i + 1) * delta for i in range(periods)]
    forecast_df = pd.DataFrame({"date": future_dates, "predicted_capacity": preds})
    return {"forecast_df": forecast_df, "model_info": {"lags": forecaster["lags"]}}

def retrieve_tool(rag: TinyRAG, query: str, top_k: int = 3) -> Dict[str, Any]:
    snippets = rag.retrieve(query, top_k=top_k)
    return {"snippets": snippets}

def generate_with_gemini(prompt: str) -> str:
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        return local_plan_generator(prompt)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    body = {"prompt": {"text": prompt}, "temperature": 0.2, "maxOutputTokens": 800}
    try:
        r = requests.post(API_URL, headers=headers, json=body, timeout=20)
        r.raise_for_status()
        resp = r.json()
        if isinstance(resp, dict):
            if "candidates" in resp and resp["candidates"]:
                text = resp["candidates"][0].get("content", "") or resp["candidates"][0].get("display", "")
                if text:
                    return text
            if "output" in resp and isinstance(resp["output"], list):
                return " ".join([str(x.get("content", "")) if isinstance(x, dict) else str(x) for x in resp["output"]])
            if "text" in resp:
                return resp["text"]
            if "content" in resp:
                return resp["content"]
        return json.dumps(resp, indent=2)[:4000]
    except Exception as e:
        st.warning(f"Calling Gemini failed: {e}. Using local fallback.")
        return local_plan_generator(prompt)

def local_plan_generator(prompt: str) -> str:
    import re
    nums = re.findall(r"\d+\.?\d*", prompt)
    top_nums = nums[:6]
    summary = "Local fallback procurement plan:\n\n"
    summary += "- Summary: Forecast indicates upcoming capacity needs.\n"
    if top_nums:
        summary += f"- Quick numbers found in prompt (first few): {', '.join(top_nums)}.\n"
    summary += "- Actions:\n"
    summary += "  1. Verify forecasts with historical trend owners.\n"
    summary += "  2. Create purchase request for spare capacity (20% buffer) for highest-risk systems.\n"
    summary += "  3. Stagger procurement across 2-3 vendors to avoid single-supplier risk.\n"
    summary += "  4. Plan lead times and logistics; prioritize items with long lead times.\n"
    summary += "  5. Re-run forecast weekly and update procurement schedule.\n\n"
    summary += "Note: set GEMINI_API_KEY to enable AI-generated, more detailed plans."
    return summary

def generate_plan_tool(forecast_df: pd.DataFrame, snippets: List[str], organization_notes: str = "") -> str:
    prompt_parts = []
    prompt_parts.append("You are an expert data-center capacity planner. Produce a short, prioritized procurement plan.")
    prompt_parts.append("Forecast (date, predicted_capacity):")
    fc_text = "\n".join([f"{row['date'].strftime('%Y-%m-%d')}: {row['predicted_capacity']:.2f}" for _, row in forecast_df.reset_index().iterrows()])
    prompt_parts.append(fc_text)
    if snippets:
        prompt_parts.append("\nRelevant documents / context snippets:")
        for s in snippets:
            prompt_parts.append(f"- {s[:800]}")
    if organization_notes:
        prompt_parts.append("\nOrganization notes:")
        prompt_parts.append(organization_notes[:1000])
    prompt_parts.append("\nProduce:\n1) Priority procurement items (short bullets). 2) Estimated quantities and timing. 3) Risks & mitigations. Keep answer ~300-500 words.")
    prompt = "\n\n".join(prompt_parts)
    return generate_with_gemini(prompt)

# ---------------- Streamlit UI ----------------

with st.sidebar:
    st.header("Upload & Settings")
    uploaded = st.file_uploader("Upload timeseries CSV (date column + capacity column). Example columns: date, capacity", type=["csv"])
    date_col = st.text_input("Date column name", value="date")
    value_col = st.text_input("Value column name", value="capacity")
    periods = st.number_input("Forecast periods (steps)", min_value=1, max_value=365, value=14)
    retrieve_k = st.slider("RAG: top-k snippets to include", 0, 10, 3)
    st.markdown("**Knowledge / RAG**: upload small .txt files or paste context below.")
    uploaded_texts = st.file_uploader("Upload text files (optional)", type=["txt","md"], accept_multiple_files=True)
    paste_context = st.text_area("Or paste knowledge/context snippets (one per line)", height=120)
    st.markdown("---")
    st.markdown("**Gemini API key** (optional): set `GEMINI_API_KEY` env var or in Streamlit secrets.")
    st.caption("If not set, the app will use a safe local fallback generator.")

if not uploaded:
    st.info("Please upload a CSV to start.")
    st.stop()

# Read CSV
try:
    df = read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Ensure datetime
try:
    df = ensure_datetime(df, date_col)
except Exception as e:
    st.error(f"Error processing date column '{date_col}': {e}")
    st.stop()

if value_col not in df.columns:
    st.error(f"Value column '{value_col}' not found in uploaded CSV columns: {list(df.columns)}")
    st.stop()

st.header("Data preview")
st.dataframe(df.tail(10))

# Build RAG index
rag = TinyRAG()
texts = []

# process uploaded files
if uploaded_texts:
    for f in uploaded_texts:
        try:
            txt = f.read().decode("utf-8")
        except:
            try:
                txt = f.read().decode("latin-1")
            except:
                txt = ""
        if txt:
            texts.append(txt)

# process pasted context
if paste_context:
    for line in paste_context.splitlines():
        if line.strip():
            texts.append(line.strip())

# add default snippet if none provided
if not texts:
    texts.append("Default knowledge snippet: Always check previous capacity trends and vendor lead times.")

rag.index_texts(texts)
st.success(f"Indexed {len(texts)} knowledge snippets for RAG retrieval.")

# Run agentic pipeline
st.header("Run pipeline")
if st.button("Run forecast + generate procurement plan"):
    with st.spinner("Running small forecast and generating plan..."):
        tool_res = forecast_tool(df, date_col, value_col, periods=int(periods))
        forecast_df = tool_res["forecast_df"]
        st.subheader("Forecast")
        st.dataframe(forecast_df)
        q_text = "Upcoming high capacity dates: " + ", ".join([f"{r['date'].strftime('%Y-%m-%d')} {r['predicted_capacity']:.0f}" for _, r in forecast_df.sort_values("predicted_capacity", ascending=False).head(3).reset_index().iterrows()])
        ret_res = retrieve_tool(rag, q_text, top_k=retrieve_k)
        snippets = ret_res["snippets"]
        if snippets:
            st.subheader("Retrieved snippets (RAG)")
            for i, s in enumerate(snippets):
                st.markdown(f"**Snippet {i+1}:** {s[:800]}{'...' if len(s) > 800 else ''}")
        plan = generate_plan_tool(forecast_df, snippets, organization_notes="")
        st.subheader("Procurement Plan (generated)")
        st.write(plan)
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download forecast CSV", data=csv, file_name="forecast.csv", mime="text/csv")

st.markdown("---")
st.caption("This app is RAG, the forecasting model, formal agent framework.")
