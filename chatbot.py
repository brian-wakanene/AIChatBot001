import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Chatbot")

def load_text_generator():
    text_generator = pipeline("text-generation", model = "gbt-2")
    text_generator.tokenizer.pad_token = text_generator.tokenizer.eos.token
    return text_generator

SYSTEM_INSTRUCTIONS = (
    "You are a helpfull assistant for software engineering"
    "Answer consisly and to the point"
    "Use markdown to format the answer"
    "Use emojis to make your answer engaging"
)

def build_conversation_prompt(chat_history, user_question):
    formated_conversation = []
    for previous_question, previous_answer in chat_history:
        formated_conversation.append(f"User: {previous_question}\nAssistant: {previous_answer}\n")

    formated_conversation.append(f"User: {user_question}\nAssistant:")
    return SYSTEM_INSTRUCTIONS + "\n" + "\n".join(formated_conversation)

st.title("Brian - ChatBot UI")
st.caption("Ask me anything you want")

#Sidebar
with st.sidebar:
    st.header("Model Controls and Configs")
    max_new_tokens = st.slider("Max new tokes", min_value = 10, value = 50, step = 10)
    temperature = st.slider("Temperature", min_value = 0.1, max_value = 1.0, value = 0.5, step = 0.1)

if st.button("Clear Chat"):
    st.session_state.chat_history = ["start new chat"]
    st.success("Chat history cleared")


#Display chat history

for user_message, ai_reply in st.session_state.chat_history:
    st.chat_message("User").write(user_message)
    st.chat_message("assistant").markdown(ai_reply)


#user input
user_input = st.chat.input("Ask me anything")
if user_input:
    st.chat_message("User").markdown(user_input)

    with st.spinner("Thinking..."):
        text_generator = load_text_generator()
        
        prompt = build_conversation_prompt(st.session_state.chat_history, user_input)
        generation_output = text_generator(
            max_new_token = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            pad_token_id = text_generator.token.eos_token_id,
            pad_token_id = text_generator.token.eos_token_id,
        )[0]['generated_text']

    #Exracting model answer
    if "Assistant:" in generation_output:
        generated_answer = generation_output.split("Assistant:")[0].strip()

#displaying and storing chatbot response

st.chat_message("assistant").markdown(generated_answer)
st.section_state.chat_history.append((user_input, generated_answer))

