# Movie Recommender System

A personalized movie recommendation engine built with Python and Streamlit, using the MovieLens 1M dataset. This project demonstrates content-based filtering techniques to suggest movies based on user preferences.

## Features

- **Movie-Based Recommendations**: Find similar movies to ones you already enjoy
- **Genre-Based Filtering**: Discover top-rated movies in your favorite genres
- **Popular Movies**: Explore critically acclaimed films with high user ratings
- **Interactive Interface**: User-friendly Streamlit interface for easy navigation

## Dataset

This project uses the MovieLens 1M dataset which includes:
- 1 million ratings from 6,000 users on 4,000 movies
- Movie metadata including titles, genres, and release years
- User demographic information

## Technology Stack

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms for content-based filtering
- **Matplotlib/Seaborn**: Data visualization tools

## Installation and Setup

1. Clone this repository
2. Install required packages:
pip install -r requirements.txt
Copy
3. Run the app:
streamlit run app/app.py

## How It Works

The recommendation system uses content-based filtering to find movies similar to a user's preferences:

1. **Data Preprocessing**: Movie features are extracted from genres and other metadata
2. **TF-IDF Vectorization**: Text features are converted into numerical vectors
3. **Cosine Similarity**: A similarity matrix is computed between all movies
4. **Recommendation Generation**: The most similar movies to a given input are identified

## Project Structure
movie_recommender/
├── app/
│   ├── app.py             # Streamlit application
│   └── init.py        # Package initialization
├── data/
│   ├── processed_*.pkl    # Preprocessed data files
│   └── init.py        # Package initialization
├── recommender/
│   ├── recommender.py     # Recommendation engine
│   └── init.py        # Package initialization
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

## Future Improvements

- Implement collaborative filtering for personalized recommendations
- Add user accounts and personal rating history
- Enhance the UI with movie posters and trailers
- Include more advanced filtering options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) for providing the dataset
- [Streamlit](https://streamlit.io/) for the web app framework
