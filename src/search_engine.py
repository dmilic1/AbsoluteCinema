import os
import joblib
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from processor import ContentProcessor


class MovieSearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.nn_model = None

    def build_and_save(self, movies_csv, credits_csv):
        print("--- Step 1: Processing Metadata ---")
        proc = ContentProcessor(movies_csv, credits_csv)
        df = proc.process()

        print("--- Step 2: Vectorizing ---")
        embeddings = self.model.encode(df['soup'].tolist(), show_progress_bar=True)

        print("--- Step 3: Fitting Nearest Neighbors ---")
        # 'cosine' metric is best for text similarity
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
        self.nn_model.fit(embeddings)

        print("--- Step 4: Saving Models ---")
        os.makedirs('../models', os.path.exists('../models') or 0o777)
        joblib.dump(self.nn_model, '../models/nn_model.joblib')
        joblib.dump(embeddings, '../models/embeddings.joblib')
        df.to_pickle('../models/metadata.pkl')
        print("Done! Files created: nn_model.joblib, embeddings.joblib, metadata.pkl")


if __name__ == "__main__":
    engine = MovieSearchEngine()
    engine.build_and_save('../data/tmdb_5000_movies.csv', '../data/tmdb_5000_credits.csv')
