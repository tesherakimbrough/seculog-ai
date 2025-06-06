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

# Chart functions
def make_event_chart(df):
    event_counts = df['event'].value_counts()
    fig = go.Bar(x=event_counts.index, y=event_counts.values)
    return fig

def make_top_ip_chart(df):
    top_ips = df['source_ip'].value_counts().nlargest(5)
    fig = go.Pie(labels=top_ips.index, values=top_ips.values)
    return fig

# App layout
st.title("🔍 SecuLog AI - Live AI-Powered Security Log Analyzer (Lite Demo)")

st.markdown("""
- LLM-based Summarization 📝  
- RAG-powered Security Q&A 🤖  
- Interactive Charts 📈  

*Note: Full Semantic Search is available in the local version of this app.*
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload Security Log (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Log Data", df.head())

    # Tabs → Lite version (3 tabs)
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
