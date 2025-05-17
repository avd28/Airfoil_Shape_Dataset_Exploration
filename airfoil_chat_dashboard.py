import streamlit as st
import pandas as pd
import sqlite3
import openai
import matplotlib.pyplot as plt
import io
import re
import os

st.set_page_config(page_title="Airfoil Chat Assistant", layout="wide")

# --- CONFIGURATION ---
# Set your OpenAI API key (or use Streamlit secrets)
openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", None))

# --- DATA LOADING ---
@st.cache_data
def load_data():
    conn = sqlite3.connect('airfoil_data.db')
    df = pd.read_sql('SELECT * FROM airfoils', conn)
    conn.close()
    return df

df = load_data()

# --- STREAMLIT UI ---
st.title("🛩️ Airfoil Database Chat Assistant")
st.write("Ask questions about the airfoil dataset in natural language. The assistant can answer and generate plots!")

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"])  # Show plot if present

# --- CHAT INPUT ---
user_input = st.chat_input("Ask me about the airfoil data!")

# --- LLM PROMPT SETUP ---
system_prompt = f"""
You are an assistant for airfoil data analysis. The dataframe columns are: {', '.join(df.columns)}.
Answer the user's question. If a plot is needed, provide Python code using pandas and matplotlib.
Return code in a markdown code block (```python ... ```). Only use the dataframe 'df'.
"""

def extract_code_blocks(text):
    # Extract python code blocks from markdown
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return [cb.strip() for cb in code_blocks]

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- LLM CALL ---
    with st.spinner("Thinking..."):
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Error: {e}"

    # --- DISPLAY ANSWER ---
    with st.chat_message("assistant"):
        st.markdown(answer)
        # --- EXECUTE AND SHOW PLOT IF CODE PRESENT ---
        code_blocks = extract_code_blocks(answer)
        for code in code_blocks:
            try:
                # Redirect matplotlib output to a buffer
                buf = io.BytesIO()
                plt.close('all')
                exec_globals = {"df": df, "plt": plt}
                exec(code, exec_globals)
                plt.savefig(buf, format='png')
                buf.seek(0)
                st.image(buf)
                st.session_state["messages"].append({"role": "assistant", "content": answer, "image": buf})
            except Exception as e:
                st.error(f"Error executing code: {e}")
                st.session_state["messages"].append({"role": "assistant", "content": answer + f"\n\nError executing code: {e}"})
                break
        else:
            st.session_state["messages"].append({"role": "assistant", "content": answer}) 