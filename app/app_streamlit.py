import streamlit as st
from retrieve_generate import retrieve, generate, State, vector_store, prompt, llm
from langchain_core.documents import Document

st.set_page_config(page_title="AMA Chatbot", page_icon="🤖")
st.title("Ask Me Anything Bot 🤖")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # List of dicts: {"user": ..., "bot": ..., "context": ...}

# Chat input box
user_input = st.chat_input("Ask a question...")

if user_input:
    # Retrieve relevant docs
    state = {"question": user_input}
    retrieved = retrieve(state)
    # Generate answer
    state.update(retrieved)
    generated = generate(state)
    answer = generated["answer"]
    # Save to history
    st.session_state["chat_history"].append({
        "user": user_input,
        "bot": answer,
        "context": [doc.page_content for doc in retrieved["context"]]
    })

# Display chat history
for chat in st.session_state["chat_history"]:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])
        # Optionally, show retrieved context for transparency
        with st.expander("Show retrieved context"):
            for i, ctx in enumerate(chat["context"]):
                st.markdown(f"**Doc {i+1}:**\n{ctx}")
