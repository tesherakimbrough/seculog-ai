import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import os

# Commented out for Streamlit Cloud Lite version:
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# Safe import of Ollama (Cloud won’t run it — local only)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Try to import OpenAI (for cloud or API-key use)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="SecuLog AI - Live Demo", layout="wide")

# === Robust Chart Functions ===

def make_event_chart(df):
    if 'event' not in df.columns or df['event'].dropna().empty:
        st.warning("No 'event' data available to plot. Please upload a CSV with an 'event' column.")
        return go.Figure()
    event_counts = df['event'].value_counts()
    fig = go.Figure([go.Bar(x=event_counts.index, y=event_counts.values)])
    fig.update_layout(title="Event Types", xaxis_title="Event", yaxis_title="Count")
    return fig

def make_top_ip_chart(df):
    if 'source_ip' not in df.columns or df['source_ip'].dropna().empty:
        st.warning("No 'source_ip' data available to plot. Please upload a CSV with a 'source_ip' column.")
        return go.Figure()
    top_ips = df['source_ip'].value_counts().nlargest(5)
    fig = go.Figure([go.Pie(labels=top_ips.index, values=top_ips.values)])
    fig.update_layout(title="Top 5 Source IPs")
    return fig

# === App Layout ===

st.title("🔍 SecuLog AI - Live AI-Powered Security Log Analyzer (Lite Demo)")

st.markdown("""
- LLM-based Summarization 📝  
- RAG-powered Security Q&A 🤖  
- Interactive Charts 📈  

*Note: Full Semantic Search is available in the local version of this app.*
""")

# Offer a sample CSV download for new users
with st.expander("Need a sample log file? Download one here."):
    st.markdown(
        "[Download sample_logs.csv](https://github.com/tesherakimbrough/seculog-ai/raw/main/data/sample_logs.csv)"
    )

# File uploader and user-friendly column debugging
uploaded_file = st.file_uploader("Upload Security Log (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Log Data", df.head())
    st.write("Columns detected in uploaded file:", list(df.columns))

    # CSV download/export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Current Log as CSV",
        data=csv,
        file_name="analyzed_logs.csv",
        mime="text/csv"
    )

    # OpenAI API setup (cloud ready)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY and OPENAI_AVAILABLE:
        OPENAI_API_KEY = st.sidebar.text_input(
            "Enter your OpenAI API Key (for LLM features):", type="password"
        )
    if OPENAI_API_KEY and OPENAI_AVAILABLE:
        openai.api_key = OPENAI_API_KEY

    # Streamlit Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Log Summarization", "🤖 RAG Q&A", "📈 Visual Analytics"])

    with tab1:
        st.subheader("📝 LLM-based Log Summarization")
        if OLLAMA_AVAILABLE:
            if st.button("Generate AI Summary (Ollama)"):
                if 'log_message' in df.columns:
                    log_lines = df['log_message'].astype(str).tolist()
                    log_text = "\n".join(log_lines[:200])
                    prompt = f"""
You are a security analyst AI. Analyze the following log entries and:
1. Summarize key events.
2. Highlight suspicious patterns.
3. Suggest next actions.

Logs:
{log_text}
"""
                    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
                    st.write("### AI-Generated Summary (Ollama):")
                    st.write(response['message']['content'])
                else:
                    st.info("Summarization requires a 'log_message' column in your CSV.")
        elif OPENAI_API_KEY and OPENAI_AVAILABLE:
            if st.button("Generate AI Summary (OpenAI)"):
                if 'log_message' in df.columns:
                    log_lines = df['log_message'].astype(str).tolist()
                    log_text = "\n".join(log_lines[:200])
                    prompt = f"""
You are a security analyst AI. Analyze the following log entries and:
1. Summarize key events.
2. Highlight suspicious patterns.
3. Suggest next actions.

Logs:
{log_text}
"""
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=512
                    )
                    st.write("### AI-Generated Summary (OpenAI):")
                    st.write(response.choices[0].message.content)
                else:
                    st.info("Summarization requires a 'log_message' column in your CSV.")
        else:
            if st.button("Show Sample AI Output (Demo Mode)"):
                st.write("### AI-Generated Summary (Sample)")
                st.markdown("""
- Detected multiple denied access attempts from `192.168.1.11`
- Possible brute-force attack pattern detected (4+ denied attempts within 10 minutes)
- Recommended Action: Review firewall rules and investigate `192.168.1.11`
""")
            st.info("LLM Summarization is not available (Ollama not installed, no OpenAI API key provided).")

    with tab2:
        st.subheader("🤖 RAG-powered Security Q&A")
        if OLLAMA_AVAILABLE:
            if 'log_message' in df.columns:
                rag_query = st.text_input("Ask a security question:")
                if st.button("Ask with Ollama"):
                    log_lines = df['log_message'].astype(str).tolist()
                    retrieved_logs = log_lines[:5]
                    rag_prompt = f"""
You are a security analyst AI. Based on the following relevant log entries, answer this question:

Question: {rag_query}

Relevant Logs:
{retrieved_logs}

Answer:
"""
                    rag_response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': rag_prompt}])
                    st.write("### AI Answer (Ollama):")
                    st.write(rag_response['message']['content'])
            else:
                st.info("RAG Q&A requires a 'log_message' column in your CSV.")
        elif OPENAI_API_KEY and OPENAI_AVAILABLE:
            if 'log_message' in df.columns:
                rag_query = st.text_input("Ask a security question:")
                if st.button("Ask with OpenAI"):
                    log_lines = df['log_message'].astype(str).tolist()
                    retrieved_logs = log_lines[:5]
                    rag_prompt = f"""
You are a security analyst AI. Based on the following relevant log entries, answer this question:

Question: {rag_query}

Relevant Logs:
{retrieved_logs}

Answer:
"""
                    rag_response = openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": rag_prompt}],
                        temperature=0.2,
                        max_tokens=512
                    )
                    st.write("### AI Answer (OpenAI):")
                    st.write(rag_response.choices[0].message.content)
            else:
                st.info("RAG Q&A requires a 'log_message' column in your CSV.")
        else:
            rag_query = st.text_input("Ask a security question:")
            if st.button("Show Sample Answer (Demo Mode)"):
                st.write("### AI Answer (Sample)")
                st.write("Yes, there are multiple denied connections from the same IP within a short timeframe, suggesting a possible brute-force attempt.")
            st.info("RAG Q&A is not available (Ollama not installed, no OpenAI API key provided).")

    with tab3:
        st.subheader("📈 Visual Analytics")
        st.write("### Event Types Chart")
        st.plotly_chart(make_event_chart(df))

        st.write("### Top 5 Source IPs Chart")
        st.plotly_chart(make_top_ip_chart(df))

