import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Llama Sentiment Analyzer", page_icon="😊")
st.title("😊 Statement Sentiment Analyzer")

# -------------------------
# Your Hugging Face Token
# -------------------------
from huggingface_hub import InferenceClient 
# Get token from secrets (NOT hardcoded!)
# -------------------------
# Get Hugging Face Token from Streamlit Secrets
# -------------------------
def get_hf_token():
    try:
        return st.secrets["HUGGINGFACE_TOKEN"]
    except Exception as e:
        st.error("""
        🔑 **Hugging Face Token Required!**
        
        Please add your token in Streamlit Secrets:
        1. Go to your app dashboard on Streamlit Cloud
        2. Click Settings → Secrets
        3. Add: `HUGGINGFACE_TOKEN = "hf_your_token_here"`
        """)
        st.stop()
# -------------------------
# Initialize the client (NOT HuggingFaceEndpoint!)
# -------------------------
@st.cache_resource
def get_client():
    try:
        token = get_hf_token()
        # This is the key fix - using InferenceClient directly
        client = InferenceClient(
            model="meta-llama/Llama-3.2-3B-Instruct",
            token=token
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize client: {str(e)}")
        return None

# Initialize client
with st.spinner("Connecting to Llama 3.2..."):
    client = get_client()

if not client:
    st.error("Could not connect to Llama. Please check your token.")
    st.stop()
else:
    st.sidebar.success("🦙 Connected to Llama 3.2!")


# -------------------------
# Prompt Templates (optimized for Llama)
# -------------------------
sentiment_template = ChatPromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Analyze the sentiment of this text. Answer with just one word: positive or negative.

Text: {text}

Sentiment:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)

main_topic_template = ChatPromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
What is the main topic of this text? Answer with just a few words.

Text: {text}

Main topic:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)

followup_template = ChatPromptTemplate.from_template(
    """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Generate one interesting follow-up question about this text.

Text: {text}

Question:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
)

parser = StrOutputParser()

# -------------------------
# Output Formatting
# -------------------------
output_formatter = RunnableLambda(lambda responses: (
    f"📝 **Statement:** {responses['statement']}\n\n"
    f"😊 **Overall sentiment:** {responses['sentiment'].strip()}\n\n"
    f"🎯 **Main topic:** {responses['main_topic'].strip()}\n\n"
    f"❓ **Followup question:** {responses['followup'].strip()}"
))

# -------------------------
# Chains
# -------------------------
sentiment_chain = sentiment_template | llm | parser
main_topic_chain = main_topic_template | llm | parser
followup_chain = followup_template | llm | parser

parallel_chain = RunnableParallel({
    "sentiment": sentiment_chain,
    "main_topic": main_topic_chain,
    "followup": followup_chain,
    "statement": RunnableLambda(lambda x: x['text'])
})

chain = parallel_chain | output_formatter

# -------------------------
# Streamlit UI
# -------------------------
st.subheader("Enter text to analyze")
user_input = st.text_area(
    "",
    placeholder="Example: I absolutely loved the new restaurant! The food was amazing and the service was excellent.",
    height=150
)

col1, col2, col3 = st.columns([1, 1, 1])  # Middle column is twice as wide as sides

with col2:  # Use the middle column
    analyze_button = st.button(
        "🔍 Analyze Statement", 
        type="primary",
        use_container_width=False  # This makes the button fill the column width
    )

if analyze_button:
    if not user_input or user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = chain.invoke({"text": user_input})
                st.success("✅ Analysis Complete!")
                st.markdown("---")
                st.markdown(result)
                
                # Show raw responses in expander (for debugging)
                with st.expander("Show raw model responses"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        sentiment = sentiment_chain.invoke(user_input)
                        st.info(f"**Sentiment raw:** {sentiment}\n")
                    with col2:
                        topic = main_topic_chain.invoke(user_input)
                        st.info(f"**Topic raw:** {topic}\n")
                    with col3:
                        follow = followup_chain.invoke(user_input)
                        st.info(f"**Question raw:** {follow}\n")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("🦙 About Llama 3.2")
    st.write("""
    **Meta's Llama 3.2 3B** is a lightweight, efficient model perfect for:
    - Sentiment analysis
    - Topic extraction
    - Question generation
    """)
    
    
    st.header("📝 Example Texts")
    st.write("Try these:")
    st.write("• *The movie was terrible, worst I've ever seen*")
    st.write("• *This product is okay, does what it's supposed to*")
    st.write("• *Absolutely fantastic experience, will come again!*")