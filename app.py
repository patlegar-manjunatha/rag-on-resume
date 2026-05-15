import streamlit as st
from main import generate_chat_response, vector_index, resume_chunks

# page config
st.set_page_config(
    page_title="Resume RAG Chatbot",
    layout="centered"
)

st.title("Resume RAG Chatbot")
st.caption("simple rag chatbot using groq + openai embeddings")

# storing chats
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("ask something about the resume...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant response
    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            try:
                response = generate_chat_response(
                    user_input=user_input,
                    chat_history=st.session_state.messages,
                    index=vector_index,
                    chunks=resume_chunks
                )
            except Exception as e:
                response = "some issue happened"
            st.markdown(response)

    # storing response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })