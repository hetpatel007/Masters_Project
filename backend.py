import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline  
import warnings
import pandas as pd
from langchain.schema import Document
import requests

warnings.filterwarnings("ignore")
__all__ = ["rag_chain", "handle_interaction"]


# Set working directory
os.chdir("/Users/hetpatel/PycharmProjects/HealthChatBot-main/End-to-end-Medical-Chatbot/research")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PINECONE_API_KEY="pcsk_5jG6sr_R5GFsYwZJ8sAQtf4LdxP1qDzHu6qyFHYqraCD38VCYU4aUDcANdRbTYEGQK6UwS"
OPENAI_API_KEY="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"
GOOGLE_API_KEY="AIzaSyBuEiAlSNCe6uBLo9awBa3T7BCJ3ZTrcqE"

llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=0.3,
        max_output_tokens=500,
        google_api_key=GOOGLE_API_KEY,
    )

# Load and split PDFs
def load_and_split_data(pdf_path="Data/", csv_dir="Data/"):
    from glob import glob
    # Load PDF documents
    loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
    pdf_documents = loader.load()

    # Load all CSV files in the directory
    csv_documents = []
    csv_files = glob(os.path.join(csv_dir, "*.csv"))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            question = str(row.get("Question", "")).strip()
            answer = str(row.get("Answer", "")).strip()
            if question and answer:
                content = f"Q: {question}\nA: {answer}"
                csv_documents.append(Document(page_content=content))

    # Combine PDF and CSV data
    all_documents = pdf_documents + csv_documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(all_documents)

# Download embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone and return vectorstore
def get_vectorstore(text_chunks, embeddings, index_name="medicalbot"):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    try:
        return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    except:
        return PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )

# Build RAG chain
def build_rag_chain():
    text_chunks = load_and_split_data()
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(text_chunks, embeddings)
    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 15})
    system_prompt = (
    "You are a helpful and empathetic AI medical assistant. "
    "Use the information below to answer the question clearly and give as much information as possible from the information below."
    "Avoid mentioning 'provided context', 'source', or 'table of contents'. "
    "If the answer is not fully available, do your best to provide a thoughtful and relevant explanation based on what you know.\n\n"
    "{context}"
)



    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Expose RAG chain
rag_chain = build_rag_chain()

# ========== Emotion Detection ==========
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def detect_emotion(text):
    emotions = emotion_detector(text)
    return emotions[0]['label']

# ========== Query Handling with Emotion ==========
def handle_interaction(query):
    if is_greeting(query):
        return "👋 Hello! I'm MediSense, your AI medical assistant. How can I help you today?", "general"
    
    intent = classify_intent(query)
    print(f"🧠 Detected Intent: {intent}")

    if intent == "general":
        final_query = f"Answer this in two lines: {query}"
        response = llm.invoke(final_query)
        ai_response = "I am a medical chatbot, I am not equipped to answer these questions but I can tell you that: \n\n" + response.content.strip()
        return ai_response, "general"
    
    else:
        emotion = detect_emotion(query)
        print(f"🧠 Detected Emotion: {emotion}")

        emotion_prompt = {
            "sadness": "Respond empathetically",
            "anger": "Respond calmly and with understanding",
            "joy": "Respond positively and excitedly",
            "fear": "Respond with calmness and reassurance",
            "love": "Respond warmly and appreciatively",
            "surprise": "Respond enthusiastically and with curiosity",
            "neutral": "Respond informatively and calmly"
        }.get(emotion, "Respond informatively and calmly")

        final_query = f"{emotion_prompt} to the following query: {query}"

        response_data = rag_chain.invoke({"input": final_query})
        ai_response = response_data.get("answer", "No response generated.")
        return ai_response, "medical"

def classify_intent(input_text):
    classification_prompt = (
        "Classify the following question as either 'medical' or 'general'. "
        "Only respond with one word: medical or general.\n\n"
        f"Question: {input_text}"
    )
    response = llm.invoke(classification_prompt)
    return response.content.strip().lower()

def is_greeting(input_text):
    prompt = (
        "Determine if the following input is a greeting (like 'hi', 'hello', 'hey', etc.). "
        "Only respond with 'yes' or 'no'.\n\n"
        f"Input: {input_text}"
    )
    response = llm.invoke(prompt)
    return response.content.strip().lower() == "yes"

def find_nearby_health_facilities(zip_code, query):
    specialty = classify_medical_specialty(query)
    try:
        geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}&key=AIzaSyD98msEePHa4TNRfOJFfiKlwN3MK6Dqj6I"
        geo_response = requests.get(geo_url).json()

        if geo_response.get("status") != "OK":
            return f"❌ Geocode failed: {geo_response.get('status', 'Unknown error')}"

        location = geo_response["results"][0]["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]

        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": 30000,
            "keyword": specialty,
            "key": "AIzaSyByyeIDSPs2BC7yIKoHaksDQP97zvqDcrE"
        }
        places_response = requests.get(places_url, params=params).json()

        results = places_response.get("results", [])
        if not results:
            return f"❌ No nearby healthcare facilities found for '{query}'."

        response = f"<b>🏥 Nearby Healthcare Facilities related to your question:</b><br><br>"
        for place in results[:5]:
            name = place["name"]
            address = place.get("vicinity", "Address not available")
            rating = place.get("rating", "No rating")
            place_id = place.get("place_id", None)

            # Create Google Maps link
            if place_id:
                maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
                link_html = f"<a href='{maps_url}' target='_blank'>View on Google Maps</a>"
            else:
                link_html = ""

            # Build the display block
            response += f"- <b>{name}</b> ⭐ {rating}<br>{address}<br>{link_html}<br><br>"

        return response

    except Exception as e:
        return f"⚠️ Error occurred: {e}"
    

def classify_medical_specialty(query):
    specialty_prompt = (
        "You are a medical assistant. Based on the user's query, classify it into the most appropriate healthcare department.\n\n"
        "Choose only from the following departments:\n"
        "- Orthopedics\n"
        "- Cardiology\n"
        "- Dentistry\n"
        "- Psychiatry\n"
        "- Dermatology\n"
        "- Ophthalmology\n"
        "- Obstetrics and Gynecology (OB-GYN)\n"
        "- Pulmonology\n"
        "- Gastroenterology\n"
        "- Neurology\n"
        "- Urology\n"
        "- Nephrology\n"
        "- Endocrinology\n"
        "- Pediatrics\n"
        "- Oncology\n"
        "- Rheumatology\n"
        "- ENT (Ear, Nose, Throat)\n"
        "- General Surgery\n"
        "- Radiology\n"
        "- Allergy and Immunology\n"
        "- Infectious Disease\n"
        "- Hematology\n"
        "- Physical Therapy\n"
        "- Emergency Medicine\n"
        "- Primary Care / General Medicine\n"
        "\n"
        "If you are unsure, default to 'Primary Care / General Medicine'.\n\n"
        "Examples:\n"
        "- 'I fractured my bone' -> Orthopedics\n"
        "- 'I'm having chest pain' -> Cardiology\n"
        "- 'I have a toothache' -> Dentistry\n"
        "- 'I feel very anxious and sad' -> Psychiatry\n"
        "- 'I have skin rashes and itching' -> Dermatology\n"
        "- 'I have blurry vision' -> Ophthalmology\n"
        "- 'I'm pregnant and need checkups' -> Obstetrics and Gynecology (OB-GYN)\n"
        "- 'I am coughing and short of breath' -> Pulmonology\n"
        "- 'I have stomach pain and digestion issues' -> Gastroenterology\n"
        "- 'I'm having frequent headaches and dizziness' -> Neurology\n"
        "- 'I have trouble urinating' -> Urology\n"
        "- 'I have kidney problems' -> Nephrology\n"
        "- 'I have diabetes' -> Endocrinology\n"
        "- 'My child has fever' -> Pediatrics\n"
        "- 'I have cancer' -> Oncology\n"
        "- 'My joints hurt a lot' -> Rheumatology\n"
        "- 'I have ear pain and sore throat' -> ENT (Ear, Nose, Throat)\n"
        "- 'I need a surgery' -> General Surgery\n"
        "- 'I need an X-ray' -> Radiology\n"
        "- 'I have severe allergies' -> Allergy and Immunology\n"
        "- 'I have an infection' -> Infectious Disease\n"
        "- 'I have blood disorders' -> Hematology\n"
        "- 'I need rehabilitation exercises' -> Physical Therapy\n"
        "- 'I have an emergency' -> Emergency Medicine\n"
        "\n"
        "Classify the following query:\n"
        f"{query}\n\n"
        "Only output the department name exactly as listed above, and nothing else."
    )
    response = llm.invoke(specialty_prompt)
    return response.content.strip()