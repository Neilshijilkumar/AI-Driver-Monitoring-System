
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

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
            return doc

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
            return doc

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
            return doc

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
            return doc

    # --------------------------------------
    # SIDEWAYS
    # --------------------------------------
    if any(
        word in q
        for word in [
            "sideways",
            "looking sideways",
            "head side",
            "left",
            "right"
        ]
    ):

        doc = get_document(
            "looking sideways"
        )

        if doc:
            return doc

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
            return doc

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
            return safe_doc

        return (
            "Drive safely and stay focused on the road."
        )

    return documents[best_index]

