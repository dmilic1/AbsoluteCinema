import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import requests


# --- Load API Key ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Page Config ---
st.set_page_config(page_title="AbsoluteCinema AI", page_icon="🍿", layout="centered")

st.title("AbsoluteCinema AI 🍿")
st.caption("Hybrid AI Agent: Local Retrieval + GPT-5.4 Nano Reasoning")

# --- movie posters ---
@st.cache_data
def get_tmdb_poster_by_id(movie_id):
    api_key = st.secrets["TMDB_API_KEY"]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": api_key}

    try:
        response = requests.get(url, params=params)
        data = response.json()

        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        print(f"Error fetching poster: {e}")

    return "https://via.placeholder.com/500x750.png?text=No+Image"


# --- Load Models (Cached) ---
@st.cache_resource
def load_assets():
    nn_model = joblib.load('models/nn_model.joblib')
    metadata = pd.read_pickle('models/metadata.pkl')
    metadata['title'] = metadata['title'].astype(object)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nn_model, metadata, embed_model


nn_model, metadata, embed_model = load_assets()


# --- Agent Class ---
class AbsoluteCinemaAgent:
    def __init__(self, client, nn_model, metadata, embed_model):
        self.client = client
        self.nn_model = nn_model
        self.metadata = metadata
        self.embed_model = embed_model
        self.chat_history = []

    def get_recommendations(self, query, n=3):
        query_vec = self.embed_model.encode([query])
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=n)

        results = []
        for i, idx in enumerate(indices[0]):
            movie = self.metadata.iloc[idx]

            # We store everything in a dictionary for maximum flexibility
            results.append({
                "id": movie['id'],  # This is your TMDb "Golden Key"
                "title": movie['title'],
                "overview": movie['overview'][:300],
                "similarity": 1 - distances[0][i],  # Convert distance to similarity score
                "metadata_text": f"TITLE: {movie['title']}\nOVERVIEW: {movie['overview'][:200]}"
            })

        return results

    def ask(self, user_input):
        context_movies = self.get_recommendations(user_input)

        system_prompt = (
            "You are AbsoluteCinema, a witty and expert movie recommender.\n"
            "ONLY recommend movies from the provided context.\n"
            "Explain WHY they match the user's request.\n"
            "Be concise and engaging.\n"
        )

        self.chat_history.append({"role": "user", "content": user_input})

        response = self.client.responses.create(
            model="gpt-5.4-nano",
            input=[
                {"role": "system", "content": system_prompt},
                *self.chat_history[-3:],
                {
                    "role": "user",
                    "content": f"""
        RELEVANT MOVIES:
        {context_movies}

        USER REQUEST:
        {user_input}

        Return exactly 3 recommendations.
        Format:
        - Title (Year): short reason
        """
                }
            ],
            max_output_tokens=200
        )

        reply = response.output[0].content[0].text
        # Access the content correctly for the Chat Completion object
        #reply = response.choices[0].message.content
        self.chat_history.append({"role": "assistant", "content": reply})

        return reply, context_movies


# --- Initialize Agent ---
if "agent" not in st.session_state:
    st.session_state.agent = AbsoluteCinemaAgent(
        client, nn_model, metadata, embed_model
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header(" System Info ")
    st.write("Model: gpt-5.4-nano")
    st.write("Retrieval: KNN + MiniLM")
    st.write("Memory: Last 3 messages")
    st.success("Grounded responses enabled")

# --- Chat UI ---
if prompt := st.chat_input("What are we watching tonight?"):

    # Safety check
    if len(prompt) > 400:
        st.error("Query too long (max 400 characters).")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Curating your movies..."):

            answer, context = st.session_state.agent.ask(prompt)

            st.markdown("## 🎬 Your Recommendations")

            # --- Render as styled cards with placeholder posters ---
            movies = [m for m in answer.split("\n") if m.strip()]

            for movie in movies:
                try:
                    title, reason = movie.split(":", 1)
                    title = re.sub(r"^[ *-]+|[ *-]+$", "", title).strip()
                    reason = re.sub(r"^\*+|\*+$", "", reason).strip()
                except ValueError:
                    title, reason = movie, ""

                # Find matching movie from context
                matched = next((m for m in context if m["title"] in title), None)

                if matched:
                    poster_url = get_tmdb_poster_by_id(matched["id"])
                else:
                    poster_url = "https://via.placeholder.com/500x750.png?text=No+Image"

                st.markdown(f"""
                    <div style="
                        display:flex;
                        background-color:#1c1f26;
                        padding:15px;
                        border-radius:15px;
                        margin-bottom:15px;
                        border:1px solid #333a45;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    ">
                        <div style="flex-shrink:0; margin-right:15px;">
                            <img src="{poster_url}" style="width:110px; height:165px; border-radius:10px; object-fit: cover;" />
                        </div>
                        <div style="flex-grow:1;">
                            <div style="font-size:20px; font-weight:bold; color:#f5c518; margin-bottom:8px;">
                                {title}
                            </div>
                            <div style="font-size:15px; color:#e5e7eb; line-height:1.4;">
                                {reason}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # --- Divider ---
            st.markdown("---")

            # --- Transparency section ---
            with st.expander("🔍 How these were chosen"):
                st.caption("Results retrieved using semantic similarity (MiniLM + KNN)")
                st.code(context, language="markdown")

            # Save message
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )