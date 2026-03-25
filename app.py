import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

# --- Load API Key ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Page Config ---
st.set_page_config(page_title="AbsoluteCinema AI", page_icon="🍿", layout="centered")

st.title("AbsoluteCinema AI 🍿")
st.caption("Hybrid AI Agent: Local Retrieval + GPT-5.4 Nano Reasoning")

# --- Dark mode
st.markdown("""
    <style>
    /* Backgrounds */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1c1f26;
    }
    /* Chat messages */
    .stChatMessage {
        background-color: #1c1f26;
        color: #f0f0f0;
    }
    /* Buttons */
    button, .stButton>button {
        background-color: #2a2f3a;
        color: #f5c518;
    }
    </style>
""", unsafe_allow_html=True)



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
            results.append(
                f"TITLE: {movie['title']}\n"
                f"OVERVIEW: {movie['overview'][:300]}\n"
                f"SIMILARITY: {distances[0][i]:.2f}"
            )

        return "\n\n".join(results)

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

                # Build placeholder poster URL
                poster_url = f"https://via.placeholder.com/100x150.png?text={title.replace(' ', '+')}"

                st.markdown(f"""
                <div style="
                    display:flex;
                    background-color:#1c1f26;
                    padding:10px;
                    border-radius:12px;
                    margin-bottom:12px;
                    border:1px solid #2a2f3a;
                    align-items:flex-start;
                ">
                    <div style="flex-shrink:0; margin-right:10px;">
                        <img src="{poster_url}" style="width:100px; height:150px; border-radius:8px;" />
                    </div>
                    <div>
                        <div style="
                            font-size:18px;
                            font-weight:bold;
                            color:#f5c518;
                        ">
                            {title}
                        </div>
                        <div style="
                            font-size:14px;
                            color:#d1d5db;
                            margin-top:5px;
                        ">
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