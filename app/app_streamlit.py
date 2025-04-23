import streamlit as st
from retrieve_generate import retrieve, generate, State, vector_store, prompt, llm
from langchain_core.documents import Document

st.set_page_config(page_title="AMA Chatbot", page_icon="🤖")
st.title("Ask Me (Chaitanya) Anything Bot 🤖")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # List of dicts: {"user": ..., "bot": ..., "context": ...}

# Display chat history on app rerun
for message in st.session_state['chat_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        # Optionally, show retrieved context for transparency
        if message['role'] == 'assistant':  # would be true only for bot responses
            with st.expander("Show retrieved context"):
                for i, ctx in enumerate(message["context"]):
                    st.markdown(f"**Doc {i+1}:**\n{ctx}")

# Chat input box
user_input = st.chat_input("Ask a question...")

# Interactive chat
if user_input:
    # Append user message to chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Retrieve relevant docs
    state = {"question": user_input}
    retrieved = retrieve(state)
    # Generate answer
    state.update(retrieved)
    generated = generate(state)
    answer = generated["answer"]

    # Append bot message to chat history
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "context": [doc.page_content for doc in retrieved["context"]]
    })

    # Display bot message in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        # Optionally, show retrieved context for transparency
        with st.expander("Show retrieved context"):
            for i, ctx in enumerate(retrieved["context"]):
                st.markdown(f"**Doc {i+1}:**\n{ctx}")