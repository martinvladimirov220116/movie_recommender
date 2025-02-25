import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recommender.recommender import MovieRecommender


# Set page config
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Add this function near the top of your file
def format_title_for_display(title):
    # Check if the title ends with ", The"
    if title.endswith(", The"):
        return "The " + title[:-5]
    # Similar handling for other articles if needed
    elif title.endswith(", A"):
        return "A " + title[:-3]
    elif title.endswith(", An"):
        return "An " + title[:-4]
    else:
        return title

# Initialize recommender
@st.cache_resource
def load_recommender():
    return MovieRecommender().prepare_content_based()

recommender = load_recommender()

# Page title
st.title("🎬 Movie Recommender System")

# Create tabs for different recommendation types
tab1, tab2, tab3 = st.tabs(["Movie-Based", "Genre-Based", "Popular Movies"])

with tab1:
    st.header("Find movies similar to ones you like")
    
    # Movie search box
    movie_input = st.text_input("Enter a movie title:")
    
    if movie_input:
        recommendations = recommender.get_movie_recommendations(movie_input)
        
        if recommendations.empty:
            st.error(f"Movie '{movie_input}' not found. Please try another title.")
        else:
            st.subheader(f"Movies similar to '{format_title_for_display(movie_input)}'")
            
            # Display recommendations in a grid
            cols = st.columns(5)
            for i, (_, row) in enumerate(recommendations.iterrows()):
                with cols[i % 5]:
                    st.subheader(format_title_for_display(row['title']))
                    if not pd.isna(row['year']):
                        st.write(f"📅 {row['year']}")
                    st.write(f"🎭 {', '.join(row['genres_list'])}")

with tab2:
    st.header("Find movies by genre")
    
    # Get unique genres
    all_genres = set()
    for genres in recommender.movies_df['genres_list']:
        all_genres.update(genres)
    
    # Genre selection
    selected_genres = st.multiselect("Select genres:", sorted(all_genres))
    
    if selected_genres:
        genre_recommendations = recommender.get_genre_recommendations(selected_genres)
        
        st.subheader(f"Top rated {', '.join(selected_genres)} movies")
        
        # Display recommendations
        for i, (_, row) in enumerate(genre_recommendations.iterrows()):
            st.write(f"**{i+1}. {format_title_for_display(row['title'])}** ({row['year']})")
            st.write(f"Rating: {row['avg_rating']:.1f}/5.0")
            st.write(f"Genres: {', '.join(row['genres_list'])}")
            st.write("---")

with tab3:
    st.header("Most Popular Movies")
    
    popular_movies = recommender.get_popular_movies(20)
    
    # Display popular movies
    for i, (_, row) in enumerate(popular_movies.iterrows()):
        st.write(f"**{i+1}. {format_title_for_display(row['title'])}** ({row['year']})")
        st.write(f"Rating: {row['avg_rating']:.1f}/5.0 (based on {row['rating_count']} ratings)")
        st.write(f"Genres: {', '.join(row['genres_list'])}")
        st.write("---")

# Add sidebar with project info
with st.sidebar:
    st.header("About")
    st.write("This movie recommender system uses the MovieLens 1M dataset to provide personalized movie recommendations.")
    st.write("Created as a portfolio project.")
    
    st.header("How it works")
    st.write("- **Movie-Based**: Finds similar movies based on content (genres)")
    st.write("- **Genre-Based**: Recommends top-rated movies in selected genres")
    st.write("- **Popular Movies**: Shows overall highest-rated movies")