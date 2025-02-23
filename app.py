import streamlit as st
import pickle
import pandas as pd

# Load movie data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
similarity = pickle.load(open('similarities.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Function to recommend movies (without posters)
def recommend(movie_name):
    try:
        movie_index = movies[movies['title'] == movie_name].index[0]  # Get movie index
        distances = similarity[movie_index]  # Get similarity scores

        # Get top 5 recommended movies (excluding the selected one)
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = [movies.iloc[i[0]].title for i in movie_list]
        return recommended_movies

    except IndexError:
        st.error("⚠️ Movie not found in the dataset. Please select a valid movie.")
        return []

# Streamlit UI
st.title("🎬 Movie Recommender System by Abhishek 🎥")

# Movie selection dropdown
selected_movie_name = st.selectbox("🔍 Choose a movie to get recommendations", movies['title'].values)

# Generate recommendations on button click
if st.button("🎯 Recommend"):
    recommendations = recommend(selected_movie_name)

    if recommendations:  # Ensure there are recommendations
        st.write("### Recommended Movies:")
        for movie in recommendations:
            st.write(f"- {movie}")  # Display only movie names
