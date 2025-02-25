import os
import pandas as pd
import numpy as np
from datetime import datetime

# Print current working directory
print(f"Working directory: {os.getcwd()}")

# Define path to data folder based on your file structure
data_dir = r"D:\Test_Projects\movie_recommender\data"

# Create paths to specific files
movies_file = os.path.join(data_dir, "movies.dat")
users_file = os.path.join(data_dir, "users.dat")
ratings_file = os.path.join(data_dir, "ratings.dat")

# Check if files exist
print(f"Data directory: {data_dir}")
print(f"Checking for data files:")
print(f"Movies file path: {movies_file}")
print(f"Movies file exists: {os.path.exists(movies_file)}")
print(f"Users file exists: {os.path.exists(users_file)}")
print(f"Ratings file exists: {os.path.exists(ratings_file)}")

# Load datasets with proper encoding and separator
print("Loading datasets...")
try:
    # Load movies data
    movies_df = pd.read_csv(
        movies_file,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movieId", "title", "genres"],
    )
    
    # Load users data
    users_df = pd.read_csv(
        users_file,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["userId", "gender", "age", "occupation", "zipCode"],
    )
    
    # Load ratings data
    ratings_df = pd.read_csv(
        ratings_file,
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    
    print("Datasets loaded successfully.")
    
    # Display initial information
    print("\nInitial dataset shapes:")
    print(f"Movies: {movies_df.shape}")
    print(f"Users: {users_df.shape}")
    print(f"Ratings: {ratings_df.shape}")
    
    # Process movies dataset
    print("\nProcessing movies dataset...")
    
    # Extract year from title and clean title
    movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)")
    movies_df["title"] = movies_df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    
    # Convert pipe-separated genres to list and create a string version for TF-IDF
    movies_df["genres_list"] = movies_df["genres"].str.split("|")
    movies_df["genres_str"] = movies_df["genres_list"].apply(lambda x: " ".join(x))
    
    # Process ratings dataset
    print("Processing ratings dataset...")
    
    # Convert timestamp to datetime
    ratings_df["timestamp"] = pd.to_datetime(ratings_df["timestamp"], unit="s")
    
    # Process users dataset
    print("Processing users dataset...")
    
    # Map age codes to age ranges
    age_mapping = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }
    users_df["age_desc"] = users_df["age"].map(age_mapping)
    
    # Map occupation codes to descriptions
    occupation_mapping = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"
    }
    users_df["occupation_desc"] = users_df["occupation"].map(occupation_mapping)
    
    # Check for missing values
    print("\nChecking for missing values:")
    print(f"Movies missing values:\n{movies_df.isnull().sum()}")
    print(f"Users missing values:\n{users_df.isnull().sum()}")
    print(f"Ratings missing values:\n{ratings_df.isnull().sum()}")
    
    # Check for duplicates
    print("\nChecking for duplicates:")
    print(f"Duplicate movie IDs: {movies_df.duplicated('movieId').sum()}")
    print(f"Duplicate user IDs: {users_df.duplicated('userId').sum()}")
    print(f"Duplicate rating entries: {ratings_df.duplicated(['userId', 'movieId']).sum()}")
    
    # Remove duplicates if any
    if movies_df.duplicated('movieId').sum() > 0:
        movies_df.drop_duplicates(subset=['movieId'], inplace=True, keep='first')
        print("Removed duplicate movie entries")
    
    # Create full dataset by merging all three
    print("\nMerging datasets...")
    ratings_movies_df = ratings_df.merge(movies_df, on="movieId")
    full_df = ratings_movies_df.merge(users_df, on="userId")
    
    print(f"Combined dataset shape: {full_df.shape}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(data_dir, "processed_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Save processed datasets
    print("\nSaving processed datasets...")
    movies_df.to_pickle(os.path.join(output_dir, "processed_movies.pkl"))
    users_df.to_pickle(os.path.join(output_dir, "processed_users.pkl"))
    ratings_df.to_pickle(os.path.join(output_dir, "processed_ratings.pkl"))
    full_df.to_pickle(os.path.join(output_dir, "movielens_full.pkl"))
    
    # Also save as CSV for easier inspection
    movies_df.to_csv(os.path.join(output_dir, "processed_movies.csv"), index=False)
    users_df.to_csv(os.path.join(output_dir, "processed_users.csv"), index=False)
    ratings_df.to_csv(os.path.join(output_dir, "processed_ratings.csv"), index=False)
    
    # Only save a sample of the full dataset as CSV (it could be very large)
    full_df.to_csv(os.path.join(output_dir, "movielens_full_sample.csv"), index=False)
    
    print("Data preprocessing completed successfully.")
    print(f"Processed files saved in: {output_dir}")
    
    # Display some summary statistics
    print("\nSummary Statistics:")
    print(f"Total unique movies: {movies_df.shape[0]}")
    print(f"Total unique users: {users_df.shape[0]}")
    print(f"Total ratings: {ratings_df.shape[0]}")
    print(f"Average rating: {ratings_df['rating'].mean():.2f}")
    print(f"Most common genres: {', '.join(movies_df['genres_str'].str.split().explode().value_counts().head(5).index)}")
    
except Exception as e:
    print(f"An error occurred during data preprocessing: {str(e)}")