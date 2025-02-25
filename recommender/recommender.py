import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommender:
    def __init__(self):
        # Get the absolute path of the current file (recommender.py)
        current_file_path = os.path.abspath(__file__)
        
        # Get the directory containing this file
        current_dir = os.path.dirname(current_file_path)
        
        # Navigate up to the project root (2 levels up from recommender directory)
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Path to processed data
        data_dir = os.path.join(project_root, "data", "processed_data")
        
        # Check if the directory exists
        if not os.path.exists(data_dir):
            # Try alternative path (might be different in deployment)
            data_dir = os.path.join(os.path.dirname(project_root), "data", "processed_data")
            
            # If still doesn't exist, try one more common structure
            if not os.path.exists(data_dir):
                data_dir = os.path.join(project_root, "data")
        
        print(f"Looking for data in: {data_dir}")
    
        # Load the data files
        try:
            self.movies_df = pd.read_pickle(os.path.join(data_dir, "processed_movies.pkl"))
            self.ratings_df = pd.read_pickle(os.path.join(data_dir, "processed_ratings.pkl"))
            self.users_df = pd.read_pickle(os.path.join(data_dir, "processed_users.pkl"))
            self.full_df = pd.read_pickle(os.path.join(data_dir, "movielens_full.pkl"))
            print("Successfully loaded data files")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in data_dir: {os.listdir(data_dir) if os.path.exists(data_dir) else 'Directory not found'}")
            raise
        
        # Initialize similarity matrix
        self.similarity_matrix = None
        
    def prepare_content_based(self):
        """Prepare the content-based filtering model"""
        # Create TF-IDF vectors from genres and other features
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['genres_str'])
        
        # Compute cosine similarity between movies
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        return self
        
    def get_movie_recommendations(self, movie_title, n=10):
        """Get content-based movie recommendations based on a movie title"""
        # Find the movie in our dataset
        movie = self.movies_df[self.movies_df['title'].str.contains(movie_title, case=False)]
        
        if movie.empty:
            return pd.DataFrame()
        
        # Get the index of the movie
        idx = movie.index[0]
        
        if self.similarity_matrix is None:
            self.prepare_content_based()
            
        # Get similarity scores for all movies
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort movies based on similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N most similar movies (excluding the input movie)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top N similar movies
        return self.movies_df.iloc[movie_indices].copy()
    
    def get_genre_recommendations(self, genres, n=10):
        """Get movie recommendations based on genres"""
        # Filter movies that contain any of the specified genres
        filtered_movies = self.movies_df[
            self.movies_df['genres_str'].str.contains('|'.join(genres), case=False)
        ]
        
        # Get average rating for each movie
        avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean().reset_index()
        avg_ratings.columns = ['movieId', 'avg_rating']
        
        # Merge with filtered movies
        movie_ratings = filtered_movies.merge(avg_ratings, on='movieId', how='left')
        
        # Sort by average rating (descending) and return top N
        return movie_ratings.sort_values('avg_rating', ascending=False).head(n)
    
    def get_popular_movies(self, n=10):
        """Get overall popular movies based on number of ratings and average rating"""
        # Count ratings for each movie
        rating_counts = self.ratings_df.groupby('movieId')['rating'].count().reset_index()
        rating_counts.columns = ['movieId', 'rating_count']
        
        # Get average rating for each movie
        avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean().reset_index()
        avg_ratings.columns = ['movieId', 'avg_rating']
        
        # Combine count and average
        movie_stats = rating_counts.merge(avg_ratings, on='movieId')
        
        # Filter to movies with at least 100 ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= 100]
        
        # Merge with movie info
        movie_data = popular_movies.merge(self.movies_df, on='movieId')
        
        # Sort by average rating (descending) and return top N
        return movie_data.sort_values('avg_rating', ascending=False).head(n)