import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Brian - ChatBot UI", layout="wide")

# Cache the model
@st.cache_resource
def load_text_generator():
    generator = pipeline("text-generation", model="gpt2")
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    return generator

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant for software engineering.\n"
    "Answer concisely and to the point.\n"
    "Use markdown to format the answer.\n"
    "Use emojis to make your answer engaging. 🤓"
)

def build_conversation_prompt(chat_history, user_question):
    messages = [SYSTEM_INSTRUCTIONS]
    for user_msg, assistant_msg in chat_history:
        messages.append(f"User: {user_msg}")
        messages.append(f"Assistant: {assistant_msg}")
    messages.append(f"User: {user_question}")
    messages.append("Assistant:")
    return "\n".join(messages)

# Title
st.title("🧠 Brian - Software Engineering ChatBot")
st.caption("Ask me anything about coding, debugging, or best practices! 🚀")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Controls")
    max_new_tokens = st.slider("Max new tokens", min_value=10, max_value=200, value=80, step=10)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for user_msg, assistant_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.markdown(assistant_msg)

# User input
if user_input := st.chat_input("Ask me anything..."):
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Thinking..."):
        text_generator = load_text_generator()
        prompt = build_conversation_prompt(st.session_state.chat_history, user_input)
        
        output = text_generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=text_generator.tokenizer.eos_token_id,
            eos_token_id=text_generator.tokenizer.eos_token_id,
            truncation=True,
            max_length=1024  # Prevent overflow
        )[0]['generated_text']

        # Extract assistant's response
        if "Assistant:" in output:
            generated_answer = output.split("Assistant:")[-1].strip()
        else:
            generated_answer = output[len(prompt):].strip()

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(generated_answer)

    # Store in history
    st.session_state.chat_history.append((user_input, generated_answer))