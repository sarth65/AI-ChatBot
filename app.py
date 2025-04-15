import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM  

# Page Config
st.set_page_config(page_title="Mistral AI Chatbot", layout="centered")

# Custom CSS for user input bar & placeholder text color
st.markdown("""
    <style>
    .stChatInput textarea {
        background-color: black !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px;
    }
    .stChatInput textarea::placeholder {
        color: #00ccff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1 style='text-align: center; color: #00ccff;'>🤖 AI Chatbot Using Ollama & Mistral</h1>", 
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define the chatbot model
model = OllamaLLM(model="mistral")

# User Input
question = st.chat_input("Ask me anything...")

if question:
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    try:
        template = "question: {question}\nAnswer = Generate the answer step by step"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        response = chain.invoke({"question": question})

        # Append AI response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    except Exception as e:
        st.error(f"Error: {e}")
