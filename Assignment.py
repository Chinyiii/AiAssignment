import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Title styling */
    .title {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }
    
    /* Header styling */
    .header {
        color: #3498db;
        font-family: 'Arial', sans-serif;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50, #3498db);
        color: white;
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Custom section colors */
    .content-based {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
    }
    
    .knowledge-based {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    .correlation {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
    }
</style>
""", unsafe_allow_html=True)

pages = {
    "Home": "🏠",
    "Content-Based Recommendations": "🔍",
    "Top 10 Recommendation based on User Preferences": "📈",
    "Game Correlation Finder": "🔗",
    "About": "ℹ️"
}

# [Keep all your existing functions exactly the same - load_content_data(), load_correlation_data(), 
# content_based_recommendations(), recommend_games()]

# Create a custom sidebar menu
st.sidebar.title("Navigation")
for page, icon in pages.items():
    if st.sidebar.button(f"{icon} {page}"):
        st.session_state.page = page

# Set the default page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Page Navigation
page = st.session_state.page

# Home Page
if page == "Home":
    st.markdown("<h1 class='title'>🎮 Welcome to the Game Recommendation System</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <h2 style='color: #3498db; text-align: center;'>Discover Your Next Favorite Game!</h2>
        <p style='text-align: center;'>
            Our Game Recommendation System helps you find games you'll love based on various methods.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Section: Content-Based Recommendations
    st.markdown("""
    <div class='card content-based'>
        <h3 class='header'>🔍 Content-Based Recommendations</h3>
        <p>
            Find games similar to the ones you already enjoy! Enter the title of your favorite game, 
            and we'll recommend similar titles based on genres, platforms, and publishers.
        </p>
        <ul>
            <li>Discover new titles that match your interests</li>
            <li>Get personalized recommendations based on your game library</li>
            <li>Easy to use with just a few clicks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Top 10 Recommendations based on User Preferences
    st.markdown("""
    <div class='card knowledge-based'>
        <h3 class='header'>📈 Top 10 Recommendations based on User Preferences</h3>
        <p>
            Upload your own game data and filter results based on your preferences. 
            Customize your search by entering preferred genres and minimum user scores.
        </p>
        <ul>
            <li>Upload your latest dataset for up-to-date recommendations</li>
            <li>Apply filters to match your taste and preferences</li>
            <li>Download your personalized recommendations as a CSV file</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Game Correlation Finder
    st.markdown("""
    <div class='card correlation'>
        <h3 class='header'>🔗 Game Correlation Finder</h3>
        <p>
            Explore how games are related based on user ratings. Select a game to see its 
            correlation with others, and find out which games share similar user reception.
        </p>
        <ul>
            <li>Identify games with similar user scores</li>
            <li>Understand game relationships through detailed correlations</li>
            <li>Discover new games based on user ratings</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 30px;'>
        <p>Use the navigation sidebar to explore different features of the app</p>
    </div>
    """, unsafe_allow_html=True)

# [Keep all your other page implementations exactly the same - Content-Based Recommendations,
# Top 10 Recommendation based on User Preferences, Game Correlation Finder, About pages]

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #eee;'>
    <p>Powered by Streamlit • Game Recommendation System</p>
</div>
""", unsafe_allow_html=True)
