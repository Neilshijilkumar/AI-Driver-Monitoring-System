
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import ollama

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ==========================================
# LOAD KNOWLEDGE BASE
# ==========================================
@st.cache_resource
def load_documents():

    with open(
        "RAG_system/knowledge_base.txt",
        "r",
        encoding="utf-8"
    ) as f:

        documents = [
            doc.strip()
            for doc in f.read().split("---")
            if doc.strip()
        ]

    return documents

documents = load_documents()

# ==========================================
# BUILD FAISS INDEX
# ==========================================
@st.cache_resource
def build_index():
    if not documents:
        raise ValueError("Knowledge base is empty")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index

index = build_index()


# ==========================================
# HELPER
# ==========================================
def get_document(keyword):

    keyword = keyword.lower()

    for doc in documents:

        first_line = doc.split("\n")[0].lower()

        if keyword in first_line:
            return doc

    return None


# ==========================================
# LLM GENERATION
# ==========================================
def generate_response(query, context):

    prompt = f"""
    You are an AI driver safety assistant.
    Use ONLY the provided safety context.

    Detected Event:
    {query}

    Safety Context:
    {context}

    Provide a concise safety recommendation in 2-3 sentences.
    Do not invent information that is not present in the context.
    """
    try:
        response = ollama.chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

        return response["message"]["content"]


    except Exception as e:
            print("Error:", e)
     
        
            return context
# ==========================================
# RETRIEVAL FUNCTION
# ==========================================
def retrieve_response(query):

    if not query:
        return (
            "Drive safely and stay focused on the road."
        )

    q = query.lower()

    # --------------------------------------
    # PHONE
    # --------------------------------------
    if any(
        word in q
        for word in [
            "phone",
            "mobile",
            "cell phone",
            "calling",
            "texting"
        ]
    ):

        doc = get_document(
            "mobile phone distraction"
        )

        if doc:
            return generate_response(
                query,
                doc) 

    # --------------------------------------
    # DRINKING
    # --------------------------------------
    if any(
        word in q
        for word in [
            "drink",
            "drinking",
            "water",
            "bottle",
            "beverage"
        ]
    ):

        doc = get_document(
            "drinking while driving"
        )

        if doc:
            return generate_response(
                query,
                doc)

    # --------------------------------------
    # DROWSINESS
    # --------------------------------------
    if any(
        word in q
        for word in [
            "drowsy",
            "sleep",
            "sleepy",
            "fatigue",
            "microsleep"
        ]
    ):

        doc = get_document(
            "drowsy driving"
        )

        if doc:
            return generate_response(
                query,doc)

    # --------------------------------------
    # HEAD DOWN
    # --------------------------------------
    if any(
        word in q
        for word in [
            "head down",
            "looking down",
            "down posture"
        ]
    ):

        doc = get_document(
            "head down posture"
        )

        if doc:
            return generate_response(
                query,doc)

    # --------------------------------------
    # SIDEWAYS
    # --------------------------------------
    if any(
        word in q
        for word in [
            "looking left",
            "looking right",
            "head turned left",
            "head turned right",
            "looking sideways",
            "sideways"
        ]
    ):

        doc = get_document(
            "looking sideways"
        )

        if doc:
            return generate_response(
                query,doc)

    # --------------------------------------
    # SAFE DRIVING
    # --------------------------------------
    if any(
        word in q
        for word in [
            "safe",
            "focused",
            "normal"
        ]
    ):

        doc = get_document(
            "safe driving"
        )

        if doc:
            return generate_response(
                query,
                doc)

    # --------------------------------------
    # FAISS FALLBACK
    # --------------------------------------
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    scores, indices = index.search(
        query_embedding,
        k=1
    )

    best_score = float(scores[0][0])
    best_index = int(indices[0][0])

    if best_score < 0.25:

        safe_doc = get_document(
            "safe driving"
        )

        if safe_doc:
            return generate_response(
                query,safe_doc)

        return (
            "Drive safely and stay focused on the road."
        )

    context = documents[best_index]

    return generate_response(
    query,
    context
)

