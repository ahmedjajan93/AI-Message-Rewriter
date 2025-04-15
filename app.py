import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
# --- Streamlit UI ---
st.set_page_config(page_title="AI Tone & Language Rewriter", layout="centered")
st.title("📝 AI Message Rewriter")
st.markdown("Transform your message into the perfect tone and language.")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Missing API Key. Please set OPENAI_API_KEY in your .env file")
    st.stop()  # This will halt the app if no key is found
os.environ["OPENAI_API_KEY"] = api_key

models = {
    "openai/gpt-3.5-turbo:free": "gpt-3.5",
    "deepseek/deepseek-chat:free": "deepseek-chat",
    "meta-llama/llama-4-maverick:free": "llama-4-maverick",
    "google/gemini-2.0-flash-thinking-exp:free": "Gemini 2.0 Flash Thinking",
}


text_input = st.text_area(
    "📨 Original Message",
    height=200,
    placeholder="Paste your message here...",
    help="Write or paste the text you want to rewrite.",
)
model_selection = st.selectbox(
    "Choose Model",
    options=list(models.keys()),
    format_func=lambda x: models[x],
    help="Choose the model for rewriting messages.",
)

# --- Optimized LLM Setup ---
llm = ChatOpenAI(
    model=model_selection,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,  # Balance creativity vs. consistency (0=strict, 1=creative)
    max_tokens=1000,  # Prevent truncated responses
)

# --- Enhanced Prompt Template ---
prompt_template = """
You are a professional communication expert. Rewrite the following message to be **clear, engaging, and perfectly match** the specified tone and language. Follow these rules:

1. **Tone**: {tone} (options: Professional, Friendly, Assertive, Polite, Casual)
2. **Language**: {language} (must be natural and idiomatic)
3. **Key Goals**:
   - Preserve the original meaning
   - Improve clarity and flow
   - Adapt to cultural norms of the target language
   - Remove redundancies
4. **Avoid**: Slang (unless Casual tone), Jargon (unless Professional), Offensive phrases.

**Original Message**:
{text}

**Rewritten Message** (only output the final version, no extra commentary):
"""

prompt = PromptTemplate(
    input_variables=["text", "tone", "language"],
    template=prompt_template,
)

chain = LLMChain(llm=llm, prompt=prompt)


col1, col2 = st.columns(2)
with col1:
    tone = st.selectbox(
        "🎭 Tone",
        ["Professional", "Friendly", "Assertive", "Polite", "Casual"],
        help="Select the desired tone for the rewritten message.",
    )
with col2:
    language = st.selectbox(
        "🌐 Language",
        [
            "English",
            "Spanish",
            "French",
            "German",
            "Chinese",
            "Hindi",
            "Arabic",
            "Japanese",
        ],
        help="Target language for translation/rewriting.",
    )

# --- Advanced Options (Collapsible) ---
with st.expander("⚙️ Advanced Settings"):
    creativity = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more predictable.",
    )
    llm.temperature = creativity  # Dynamically adjust creativity

if st.button("✨ Rewrite Message", type="primary"):
    if not text_input.strip():
        st.warning("Please enter a message to rewrite.")
    else:
        with st.spinner("Crafting the perfect version..."):
            try:
                rewritten = chain.run(
                    text=text_input,
                    tone=tone,
                    language=language,
                )
               if not result or result.strip() == "":
                     st.error("⚠️ The model returned no content. Try again later or switch models.")
              else:
                    st.markdown("### ✨ Enhanced Message")
                    st.success(result.strip())
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
