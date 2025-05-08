import streamlit as st
from backend import rag_chain, handle_interaction, find_nearby_health_facilities, classify_medical_specialty
import os
from PIL import Image
PINECONE_API_KEY="pcsk_5jG6sr_R5GFsYwZJ8sAQtf4LdxP1qDzHu6qyFHYqraCD38VCYU4aUDcANdRbTYEGQK6UwS"
OPENAI_API_KEY="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"
GOOGLE_API_KEY="AIzaSyBuEiAlSNCe6uBLo9awBa3T7BCJ3ZTrcqE"
# Page configuration
st.set_page_config(page_title="MediSense - Health Chatbot", layout="centered", initial_sidebar_state="collapsed")

# Load and center the logo image
logo = Image.open("MediSense_Logo.jpeg")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=340)

# Custom Styles
st.markdown(
    """
    <style>
    .stApp { background-color: #f4f6f8; font-family: 'Segoe UI', sans-serif; }
    .main-title { font-size: 2.4rem; color: #1a1a1a; font-weight: 700; text-align: center; }
    .subheading { font-size: 1.2rem; color: #444444; text-align: center; margin-bottom: 1.5rem; }
    div.stButton > button { background-color: #007BFF; color: white; font-size: 16px; border-radius: 12px; padding: 10px 20px; }
    div.stButton > button:hover { background-color: #0056b3; }
    .footer-note { font-size: 0.9rem; color: #555; text-align: center; margin-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# Session Initialization
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Home Page
if st.session_state.page == "Home":
    st.markdown('<div class="main-title">Welcome to MediSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheading">Your AI Health Companion for Symptom-based Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='color: black; font-size: 1rem; text-align: center;'>
        <strong>✨ Features:</strong><br>
        • 💬 AI-driven medical Q&A using real documents<br>
        • 🔗 Powered by <strong>Gemini Pro + LangChain</strong><br>
        • 🧠 Emotion-aware responses<br>
        • ⚠️ <em>This tool does not replace real medical consultation</em>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        if st.button("🚀 Start Chatbot"):
            st.session_state.page = "Chatbot"
            st.rerun()

# Chatbot Page
elif st.session_state.page == "Chatbot":
    st.markdown('<div class="main-title">MediSense Chatbot</div>', unsafe_allow_html=True)

    if "zip_code" not in st.session_state:
        st.session_state.zip_code = None

    if st.session_state.zip_code is None:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='background-color: white; padding: 30px; border-radius: 12px; box-shadow: 0px 0px 15px rgba(0,0,0,0.3); text-align: center; margin-top: 100px;'>
                    <h3 style='color: #007BFF;'>Enter Your ZIP Code</h3>
                    <p style='color: #444;'>We will find nearby healthcare facilities for you!</p>
                </div>
                """, unsafe_allow_html=True)
                zip_code = st.text_input("Enter ZIP Code", max_chars=10, label_visibility="collapsed", key="zip_input_field")
                if zip_code:
                    if zip_code.isdigit() and len(zip_code) == 5:
                        st.session_state.zip_code = zip_code
                        st.success("✅ ZIP Code saved!")
                        st.rerun()
                    else:
                        st.error("❗ Please enter a valid 5-digit ZIP code.")
        st.stop()

    # Display chat messages
    for message in st.session_state.messages:
        icon = "🧍" if message["role"] == "user" else "🤖"
        bubble_color = "#dfe9f3" if message["role"] == "user" else "#ffffff"
        
        if message["role"] == "assistant" and message.get("is_markdown", False):
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: {bubble_color}; padding: 10px 15px; border-radius: 12px; margin: 10px 0; display: flex; align-items: center; max-width: 90%;'>
                <span style='font-size: 20px; margin-right: 10px;'>{icon}</span>
                <span style='font-size: 16px; color: black;'>{message["content"]}</span>
            </div>
            """, unsafe_allow_html=True)

    if prompt := st.chat_input("Describe your symptoms or search (e.g., 'dentist', 'pharmacy')..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("🤖 Analyzing..."):
            try:
                answer_text, intent = handle_interaction(prompt)
                if intent == "medical":
                    places = find_nearby_health_facilities(st.session_state.zip_code, prompt)
                else:
                    places = ""
            except Exception as e:
                answer_text = "⚠️ Sorry, something went wrong. Please try again."
                places = ""

        # 👉 Add spacing and bold for facility section
        styled_answer = f"<div style='color: black; font-size: 16px;'>{answer_text}<br><br><b>{places}</b></div>"
        st.session_state.messages.append({"role": "assistant", "content": styled_answer, "is_markdown": True})

        st.rerun()


    st.markdown('<div class="footer-note">🩺 This chatbot is for educational purposes only and does not offer medical advice.</div>', unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns([1, 2, 1])
    with col5:
        if st.button("🔙 Back to Home"):
            st.session_state.page = "Home"
            st.session_state.messages = []
            st.session_state.zip_code = None
            st.rerun()
