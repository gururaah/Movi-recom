import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
movies = pd.read_csv('movies.csv')
credit = pd.read_csv('credits.csv')

# Merge datasets on 'title'
movies = movies.merge(credit, on='title')

# Keep relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing values
movies.dropna(inplace=True)

# Function to convert JSON-like string columns
def convert(x):
    try:
        l = [i['name'] for i in ast.literal_eval(x)]
        return l
    except (ValueError, SyntaxError):
        return []  # Return empty list if conversion fails

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Function to extract top 3 cast members
def convert_cast(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)[:3]]
    except (ValueError, SyntaxError):
        return []

movies['cast'] = movies['cast'].apply(convert_cast)

# Function to extract director from crew
def fetch_director(x):
    try:
        for i in ast.literal_eval(x):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except (ValueError, SyntaxError):
        return []

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview into list
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Remove spaces in names to create single-word tags
for col in ['cast', 'crew', 'genres', 'keywords']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

# Create a 'tags' column
movies['tags'] = movies['cast'] + movies['crew'] + movies['genres'] + movies['keywords'] + movies['overview']

# Create a new DataFrame with relevant columns
new_df = movies[['movie_id', 'title', 'tags']].copy()

# Convert list to string and lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x).lower())

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to apply stemming
def stem(text):
    return ' '.join([ps.stem(word) for word in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)

# Apply CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity
similarities = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):
    if movie_name not in new_df['title'].values:
        print("Movie not found. Please try another title.")
        return
    
    movie_index = new_df[new_df['title'] == movie_name].index[0]
    distances = similarities[movie_index]

    # Get top 5 similar movies
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print(f"Movies similar to '{movie_name}':")
    for i in movie_list:
        print(new_df.iloc[i[0]]['title'])

# Test recommendation
recommend("House of D")

# Save the model
#pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarities, open('similarities.pkl', 'wb'))
