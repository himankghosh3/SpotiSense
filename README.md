# 🎵 SpotiSense — Music Analysis, Recommendation & Popularity Prediction

A complete end-to-end data science project that analyzes 113,999 Spotify tracks to uncover patterns in music, group songs by sound, recommend similar tracks, and predict popularity using machine learning.

---

## 📌 Project Overview

SpotiSense uses Spotify's audio features — danceability, energy, valence, tempo, acousticness and more — to answer five real questions about music:

- What does the popularity landscape of Spotify look like?
- Which genres dominate and what makes them sound the way they do?
- Can a machine group songs by how they sound without being told the genre?
- Given a song you like, can we find 5 similar ones using audio features alone?
- Can we predict how popular a song will be based on its audio fingerprint?

The project is built in a structured, modular workflow across Jupyter Notebooks and deployed as an interactive Streamlit web application.

---

## 🚀 Live App

Run the app locally:

```bash
python -m streamlit run app.py
```

The app allows you to:
- Search any song by name
- View its full audio feature profile
- Get 5 similar song recommendations
- See its predicted vs actual popularity score

---

## 🗂️ Project Structure

```
SpotiSense/
│
├── Data/
│   └── cleaned_tracks.csv          # Processed dataset
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_genre_trends.ipynb        # Genre & Popularity Trends
│   ├── 03_clustering.ipynb          # KMeans Clustering + PCA
│   ├── 04_recommendation.ipynb      # Cosine Similarity Recommender
│   └── 05_ml_model.ipynb            # Random Forest Popularity Predictor
│
├── outputs/                         # Saved visualizations
│   ├── popularity_distribution.png
│   ├── top_genres_popularity.png
│   ├── correlation_heatmap.png
│   ├── popularity_correlations.png
│   ├── genre_popularity_comparison.png
│   ├── audio_features_by_genre.png
│   ├── radar_chart.png
│   ├── elbow_curve.png
│   ├── clusters_pca.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
│
├── app.py                           # Streamlit web application
├── requirements.txt                 # Dependencies
└── README.md
```

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Algorithms | KMeans, PCA, Random Forest, Cosine Similarity |
| Web App | Streamlit |
| Dataset | Spotify Tracks Dataset — Kaggle (114,000 tracks, 114 genres) |

---

## 📊 Modules

### Module 1 — Exploratory Data Analysis
Loaded and cleaned 114,000 tracks. Explored popularity distribution, audio feature correlations, and identified a large spike of zero-popularity tracks that would later impact model performance.

### Module 2 — Genre & Popularity Trends
Compared top and bottom genres by average popularity. Discovered that platform demographics — not musical quality — largely determine which genres dominate Spotify. Built radar charts to profile the audio fingerprint of popular vs unpopular genres.

### Module 3 — Clustering
Applied KMeans clustering with the Elbow Method to identify 8 natural sound categories across all tracks without using genre labels. Visualized clusters using PCA dimensionality reduction. The algorithm independently discovered categories matching real-world music classifications — acoustic, electronic, hip-hop, spoken word and more.

### Module 4 — Recommendation System
Built a content-based recommendation engine using cosine similarity on audio features. Given any song, the system returns the 5 most sonically similar tracks from the dataset — transcending genre and cultural boundaries.

### Module 5 — ML Model
Trained a Random Forest Regressor to predict popularity from audio features. Achieved an R² score of 0.54 and RMSE of 15.12. Feature importance analysis revealed that no single audio feature dominates popularity prediction — all features contribute roughly equally.

---

## 🔍 Key Findings

**1. Popularity is heavily top-skewed**
The vast majority of tracks on Spotify go undiscovered. Only a tiny fraction achieve mainstream popularity — reflecting the platform's demographic reality rather than any measure of musical quality.

**2. Platform demographics drive genre popularity**
Pop-film, K-Pop and Indian music lead popularity charts because Spotify's user base is predominantly young and global. Genres like Classical and Jazz — despite their cultural significance — rank among the lowest because their core audience is underrepresented on the platform.

**3. Music naturally clusters into 8 sound categories**
Without any genre labels, KMeans clustering independently discovered: Happy Dance Pop, High Energy Electronic, Soft Acoustic, Electronic Instrumental, Live Performance, Quiet Instrumental, Hip Hop/R&B, and Spoken Word.

**4. Audio similarity transcends culture and genre**
The recommendation system groups a Chinese Mandopop ballad, an Indian song, and a Bluegrass track together — purely because they share the same soft acoustic sound profile. Audio features know no borders.

**5. Popularity cannot be reduced to audio features alone**
Both linear correlation analysis and Random Forest feature importance confirm the same finding: no audio feature strongly predicts popularity. The model explains 54% of variance — the remaining 46% is driven by artist fame, marketing, cultural moments, and virality that audio data simply cannot capture.

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/SpotiSense.git
cd SpotiSense
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

**3. Download the dataset**

Download the Spotify Tracks Dataset from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place `dataset.csv` (renamed to `tracks.csv`) inside the `Data/` folder.

**4. Run the notebooks in order**
```
01_eda.ipynb → 02_genre_trends.ipynb → 03_clustering.ipynb → 04_recommendation.ipynb → 05_ml_model.ipynb
```

**5. Launch the app**
```bash
python -m streamlit run app.py
```

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Features | 9 audio features |
| Training Set | 91,199 tracks (80%) |
| Test Set | 22,800 tracks (20%) |
| RMSE | 15.12 |
| R² Score | 0.54 |

> **Note:** The model predicts popularity based on audio features only. Cultural factors such as artist fame, playlist placement, and social media virality are not captured and account for the remaining unexplained variance.

---

## 📁 Dataset

**Spotify Tracks Dataset** by Maharshi Pandya
- Source: [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- 114,000 tracks across 114 genres
- 21 features including audio characteristics and metadata

---

*Built as a portfolio data science project demonstrating end-to-end workflow: data cleaning, EDA, unsupervised learning, recommendation systems, supervised learning, and app deployment.*
