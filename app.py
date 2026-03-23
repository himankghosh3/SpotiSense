import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="SpotiSense",
    page_icon="",
    layout="wide"
)

# Spotify themed CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
    }
    h1, h2, h3 {
        color: #1DB954 !important;
    }
    .stTextInput > div > div > input {
        background-color: #282828;
        color: white;
        border: 2px solid #1DB954;
        border-radius: 20px;
        padding: 10px;
    }
    .stDataFrame {
        background-color: #282828;
    }
    p, label, .stMarkdown {
        color: white !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #282828 !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1DB954 !important;
    }        
    </style>
""", unsafe_allow_html=True)

# Title
st.title(" SpotiSense")
st.markdown("### Music Recommendation & Popularity Prediction")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("Data/cleaned_tracks.csv")
    return df

@st.cache_resource
def train_model(df):
    features = ["danceability", "energy", "loudness", "speechiness",
                "acousticness", "instrumentalness", "liveness",
                "valence", "tempo"]
    X = df[features]
    y = df["popularity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, features

df = load_data()
model, features = train_model(df)
X = df[features].values

with st.sidebar:
    st.markdown("##  SpotiSense")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("SpotiSense analyzes Spotify audio features to recommend similar songs and predict popularity.")
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(f" **{len(df):,} tracks**")
    st.markdown(f" **{df['track_genre'].nunique()} genres**")
    st.markdown(f" **{df['cluster_name'].nunique()} sound categories**")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1.  Search any song")
    st.markdown("2.  View audio features")
    st.markdown("3.  Get recommendations")
    st.markdown("4.  See popularity prediction")

# Search bar
st.markdown("##  Search for a Song")
song_input = st.text_input("Enter a song name", placeholder="e.g. Blinding Lights, Feels, Shape of You...")

if song_input:
    matches = df[df["track_name"].str.lower() == song_input.lower()]
    
    if matches.empty:
        st.error(f" Song '{song_input}' not found in dataset")
    else:
        song = matches.loc[matches["popularity"].idxmax()]
        
        # Song details
        st.markdown("---")
        st.markdown(f"##  {song['track_name']}")
        st.markdown(f"**Artist:** {song['artists']}")
        st.markdown(f"**Genre:** {song['track_genre']}")
        st.markdown(f"**Sound Category:** {song['cluster_name']}")
        
        # Audio features
        st.markdown("###  Audio Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Danceability", f"{song['danceability']:.2f}")
            st.metric("Energy", f"{song['energy']:.2f}")
            st.metric("Valence", f"{song['valence']:.2f}")
        with col2:
            st.metric("Acousticness", f"{song['acousticness']:.2f}")
            st.metric("Speechiness", f"{song['speechiness']:.2f}")
            st.metric("Instrumentalness", f"{song['instrumentalness']:.2f}")
        with col3:
            st.metric("Tempo", f"{song['tempo']:.0f} BPM")
            st.metric("Loudness", f"{song['loudness']:.1f} dB")
            st.metric("Liveness", f"{song['liveness']:.2f}")

# Recommendations
        st.markdown("---")
        st.markdown("###  Similar Songs You Might Like")
        
        song_idx = matches["popularity"].idxmax()
        song_vector = X[song_idx].reshape(1, -1)
        similarities = cosine_similarity(song_vector, X)[0]
        similar_indices = similarities.argsort()[::-1][1:12]
        
        recommendations = df.iloc[similar_indices][["track_name", "artists", "track_genre", "cluster_name", "popularity"]].drop_duplicates(subset=["track_name"]).head(5)
        recommendations.columns = ["Track Name", "Artist", "Genre", "Sound Category", "Popularity"]
        recommendations = recommendations.reset_index(drop=True)
        recommendations.index += 1
        
        st.dataframe(recommendations, use_container_width=True)

# Popularity Prediction
        st.markdown("---")
        st.markdown("###  Popularity Prediction")
        
        song_features = df.loc[song_idx, features].values.reshape(1, -1)
        predicted_popularity = model.predict(song_features)[0]
        actual_popularity = song["popularity"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label=" Predicted Popularity",
                value=f"{predicted_popularity:.0f} / 100"
            )
        with col2:
            st.metric(
                label=" Actual Popularity",
                value=f"{actual_popularity} / 100"
            )
        
        # Popularity bar
        st.markdown("**Predicted Popularity Score:**")
        st.caption(" Prediction based on audio features only. Cultural factors like artist fame and virality are not captured.")
        st.progress(int(predicted_popularity))