import streamlit as st
import pandas as pd
import plotly.graph_objs as go

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

    # Streamlit Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Log Summarization", "🤖 RAG Q&A", "📈 Visual Analytics"])

    with tab1:
        st.subheader("📝 LLM-based Log Summarization")
        if not OLLAMA_AVAILABLE:
            st.info("LLM Summarization is not available (ollama not installed).")
        else:
            if st.button("Generate AI Summary"):
                if 'log_message' in df.columns:
                    log_lines = df['log_message'].astype(str).tolist()
                    log_text = "\n".join(log_lines[:200])  # limit for demo
                    prompt = f"""
You are a security analyst AI. Analyze the following log entries and:
1. Summarize key events.
2. Highlight suspicious patterns.
3. Suggest next actions.

Logs:
{log_text}
"""
                    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
                    st.write("### AI-Generated Summary:")
                    st.write(response['message']['content'])
                else:
                    st.info("Summarization requires a 'log_message' column in your CSV.")

    with tab2:
        st.subheader("🤖 RAG-powered Security Q&A")
        if not OLLAMA_AVAILABLE:
            st.info("RAG Q&A is not available (ollama not installed).")
        else:
            if 'log_message' in df.columns:
                log_lines = df['log_message'].astype(str).tolist()
                rag_query = st.text_input("Ask a security question:")
                if rag_query:
                    # Simple RAG for demo → return first 5 log lines as context
                    retrieved_logs = log_lines[:5]
                    rag_prompt = f"""
You are a security analyst AI. Based on the following relevant log entries, answer this question:

Question: {rag_query}

Relevant Logs:
{retrieved_logs}

Answer:
"""
                    rag_response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': rag_prompt}])
                    st.write("### AI Answer:")
                    st.write(rag_response['message']['content'])
            else:
                st.info("RAG Q&A requires a 'log_message' column in your CSV.")

    with tab3:
        st.subheader("📈 Visual Analytics")
        st.write("### Event Types Chart")
        st.plotly_chart(make_event_chart(df))

        st.write("### Top 5 Source IPs Chart")
        st.plotly_chart(make_top_ip_chart(df))

else:
    st.info("Upload a CSV file to begin analysis. For demo, use the [sample log file](https://github.com/tesherakimbrough/seculog-ai/raw/main/data/sample_logs.csv).")
