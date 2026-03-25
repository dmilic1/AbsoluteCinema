import pandas as pd
import ast

class ContentProcessor:
    def __init__(self, movies_path, credits_path):
        self.movies_df = pd.read_csv(movies_path)
        self.credits_df = pd.read_csv(credits_path)

    def _convert_json(self, obj):
        try:
            items = ast.literal_eval(obj)
            return [i['name'] for i in items]
        except (ValueError, SyntaxError):
            return []

    def _get_director(self, crew_obj):
        try:
            crew = ast.literal_eval(crew_obj)
            for i in crew:
                if i['job'] == 'Director':
                    return i['name']
        except (ValueError, SyntaxError):
            return ""
        return ""

    def _get_top_cast(self, cast_obj, limit=3):
        try:
            cast = ast.literal_eval(cast_obj)
            return [i['name'] for i in cast[:limit]]
        except (ValueError, SyntaxError):
            return []

    def process(self):
        credits_clean = self.credits_df[['movie_id', 'cast', 'crew']]
        df = self.movies_df.merge(credits_clean, left_on='id', right_on='movie_id')

        df['genres'] = df['genres'].apply(self._convert_json)
        df['keywords'] = df['keywords'].apply(self._convert_json)
        df['cast'] = df['cast'].apply(self._get_top_cast)
        df['director'] = df['crew'].apply(self._get_director)

        def clean_tags(x):
            if isinstance(x, list):
                return [str.lower(i.replace(" ", "")) for i in x]
            return str.lower(x.replace(" ", "")) if isinstance(x, str) else ""

        for col in ['cast', 'keywords', 'genres', 'director']:
            df[col] = df[col].apply(clean_tags)

        # Create the Metadata Soup
        df['soup'] = df.apply(lambda x:
                              f"{' '.join(x['keywords'])} {' '.join(x['cast'])} {x['director']} "
                              f"{' '.join(x['genres'])} {x['overview']}", axis=1)

        # --- THE FIX: CAST TO OBJECT ---
        # Forces compatibility with Scikit-Learn slicing
        df['soup'] = df['soup'].fillna('').astype(object)
        df['title'] = df['title'].astype(object)

        return df[['id', 'title', 'soup', 'overview']]


if __name__ == "__main__":
    # Quick test
    processor = ContentProcessor('../data/tmdb_5000_movies.csv', '../data/tmdb_5000_credits.csv')
    processed_df = processor.process()
    print(f"Processed {len(processed_df)} movies.")
    print(f"Sample Soup: {processed_df['soup'].iloc[0][:100]}...")