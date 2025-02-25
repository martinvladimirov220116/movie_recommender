import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommender:
    def __init__(self):
        # Load processed data
        data_dir = r"D:\Test_Projects\movie_recommender\data\processed_data"
        self.movies_df = pd.read_pickle(os.path.join(data_dir, "processed_movies.pkl"))
        self.ratings_df = pd.read_pickle(os.path.join(data_dir, "processed_ratings.pkl"))
        self.users_df = pd.read_pickle(os.path.join(data_dir, "processed_users.pkl"))
        self.full_df = pd.read_pickle(os.path.join(data_dir, "movielens_full.pkl"))
        
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