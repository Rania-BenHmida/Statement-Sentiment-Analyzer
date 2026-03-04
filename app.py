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
HUGGINGFACE_TOKEN = "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"

# -------------------------
# Load Llama 3.2 3B (the one that works!)
# -------------------------
@st.cache_resource
def load_llama():
    try:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            huggingfacehub_api_token=HUGGINGFACE_TOKEN,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )
        return ChatHuggingFace(llm=llm)
    except Exception as e:
        st.error(f"Failed to load Llama: {str(e)}")
        return None

# Load the model
with st.spinner("Loading Llama 3.2..."):
    llm = load_llama()

if llm:
    st.sidebar.success("🦙 Llama 3.2 loaded successfully!")
else:
    st.error("Could not load Llama. Please check your token.")
    st.stop()

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