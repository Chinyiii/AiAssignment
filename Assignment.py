import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config with custom theme
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main content styling */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Title styling */
    .title-text {
        color: #2c3e50;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Card styling for sections */
    .custom-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 25px;
        border-left: 5px solid #3498db;
    }
    
    /* Section header styling */
    .section-header {
        color: #3498db;
        font-family: 'Poppins', sans-serif;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50, #34495e);
    }
    
    .sidebar .sidebar-content {
        color: white;
    }
    
    /* Custom section colors */
    .content-card {
        border-left-color: #3498db;
        background: linear-gradient(to right, #f8f9fa, #e3f2fd);
    }
    
    .knowledge-card {
        border-left-color: #2ecc71;
        background: linear-gradient(to right, #f8f9fa, #e8f5e9);
    }
    
    .correlation-card {
        border-left-color: #e74c3c;
        background: linear-gradient(to right, #f8f9fa, #ffebee);
    }
    
    /* Table styling */
    .dataframe {
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #7f8c8d;
        border-top: 1px solid #ecf0f1;
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

# [Keep all your existing data loading and recommendation functions exactly the same]
# Load the data for content-based recommendations
@st.cache_data
def load_content_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df = df.dropna(subset=['Genres', 'Platforms', 'Publisher', 'User Score'])  # Drop rows with essential missing values
    df['User Score'] = df['User Score'].astype(float)  # Ensure correct data type for user score
    df['content'] = df['Genres'] + ' ' + df['Platforms'] + ' ' + df['Publisher']
    return df

# Load the data for correlation finder
@st.cache_data
def load_correlation_data():
    path = 'all_video_games(cleaned).csv'
    df = pd.read_csv(path)
    path_user = 'User_Dataset.csv'
    userset = pd.read_csv(path_user)
    data = pd.merge(df, userset, on='Title').dropna()  
    return data

df_content = load_content_data()
df_corr = load_correlation_data()

# Function to recommend games based on cosine similarity
def content_based_recommendations(game_name, num_recommendations=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    content_matrix = vectorizer.fit_transform(df_content['content'])

    try:
        cosine_sim = cosine_similarity(content_matrix, content_matrix)
        idx = df_content[df_content['Title'].str.lower() == game_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        return df_content.iloc[sim_indices][['Title', 'Genres', 'User Score', 'Platforms', 'Release Date']]
    except IndexError:
        return pd.DataFrame(columns=['Title', 'Genres', 'User Score'])

# Function to recommend games based on file upload and filters
def recommend_games(df, preferences):
    genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
    score_filter = df['User Score'] >= preferences['Minimum User Score']
    filtered_df = df[genre_filter & score_filter]
    return filtered_df
    
# Create a custom sidebar menu with improved styling
st.sidebar.markdown("""
<div style='padding: 10px; border-radius: 8px; background: #2c3e50; margin-bottom: 20px;'>
    <h2 style='color: white; text-align: center;'>Navigation</h2>
</div>
""", unsafe_allow_html=True)

for page, icon in pages.items():
    if st.sidebar.button(f"{icon} {page}", key=f"btn_{page}"):
        st.session_state.page = page

# Set the default page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Page Navigation
page = st.session_state.page

# Home Page
if page == "Home":
    st.markdown("<h1 class='title-text'>🎮 Welcome to the Game Recommendation System</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='custom-card' style='text-align: center; background: linear-gradient(135deg, #f5f7fa, #e3f2fd);'>
        <h2 style='color: #3498db;'>Discover Your Next Favorite Game!</h2>
        <p style='font-size: 16px;'>
            Our advanced recommendation system helps you find games you'll love through multiple intelligent methods.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Section: Content-Based Recommendations
    st.markdown("""
    <div class='custom-card content-card'>
        <h3 class='section-header'>🔍 Content-Based Recommendations</h3>
        <p>
            Find games similar to your favorites by analyzing game features like genres, platforms, and publishers.
        </p>
        <ul style='padding-left: 20px;'>
            <li>Personalized recommendations based on game characteristics</li>
            <li>No user history required - works for new users</li>
            <li>Great for discovering similar gaming experiences</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Top 10 Recommendations based on User Preferences
    st.markdown("""
    <div class='custom-card knowledge-card'>
        <h3 class='section-header'>📈 Top 10 Recommendations</h3>
        <p>
            Get personalized suggestions based on your specific preferences and filters.
        </p>
        <ul style='padding-left: 20px;'>
            <li>Filter by genre, score, and other attributes</li>
            <li>Upload your own game data for customized results</li>
            <li>Export your recommendations for later reference</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Game Correlation Finder
    st.markdown("""
    <div class='custom-card correlation-card'>
        <h3 class='section-header'>🔗 Game Correlation Finder</h3>
        <p>
            Discover how games relate to each other based on user ratings and preferences.
        </p>
        <ul style='padding-left: 20px;'>
            <li>Find games with similar user appeal</li>
            <li>See detailed correlation metrics</li>
            <li>Identify hidden connections between games</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 30px; color: #7f8c8d;'>
        <p>Select a feature from the sidebar to begin exploring</p>
    </div>
    """, unsafe_allow_html=True)

# [Keep all your other page implementations exactly the same - just the styling is enhanced]
# Page 1: Content-Based Recommendations
elif page == "Content-Based Recommendations":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Find Games Similar to Your Favorite</h2>", unsafe_allow_html=True)
    st.write("This app helps you find games similar to the ones you like. Enter the game title below to get recommendations.")

    # Add a selectbox for game selection
    game_list = df_content['Title'].unique()
    game_input = st.selectbox("Choose a game from the list:", game_list)

    # Filters within the main page
    st.subheader("Filters")
    num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

    # Game information display
    if game_input:
        game_info = df_content[df_content['Title'] == game_input].iloc[0]
        st.markdown(f"### Selected Game: {game_info['Title']}")
        st.write(f"Genres: {game_info['Genres']}")
        st.write(f"Platforms: {game_info['Platforms']}")
        st.write(f"Publisher: {game_info['Publisher']}")
        st.write(f"User Score: {game_info['User Score']}")
        st.write(f"Release Date: {game_info['Release Date']}")

    # Button to get recommendations
    if st.button('Get Recommendations'):
        recommendations = content_based_recommendations(game_input, num_recommendations)
        if not recommendations.empty:
            st.markdown(f"### Games similar to {game_input}:")
            st.table(recommendations)
        else:
            st.write("No matching game found. Please try another.")

# Page 2: File Upload and Filters
elif page == "Top 10 Recommendation based on User Preferences":
    st.title("📂 Upload Latest Dataset")
    st.markdown("""
    Follow these steps to get personalized game recommendations:
    1. Upload your game data in CSV format(Ensure that the dataset is the latest).
    2. Enter your preferred genre and minimum user score.
    3. Click "Get Recommendations" to view the top suggestions.
    """)

    # Upload section
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load the dataset
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Data processing
            df_uploaded['Genres'] = df_uploaded['Genres'].astype(str).fillna('')
            df_uploaded['User Score'] = pd.to_numeric(df_uploaded['User Score'], errors='coerce')

            # Display dataset preview
            st.write("### Dataset Preview")
            st.dataframe(df_uploaded.head())

            # Filter options section
            st.subheader("🎯 Filter Options")

            genres = st.text_input(
                "Preferred Genre (e.g., Action, Adventure):",
                value="",
                placeholder="Enter genres (e.g., Action, Adventure)",
                help="Enter the genre(s) you like. Use commas for multiple genres."
            ).strip()

            min_user_score_str = st.text_input(
                "Minimum Acceptable User Score (0.0 to 10.0):",
                value="0.0",
                placeholder="Enter a score between 0.0 and 10.0",
                help="Enter a number between 0.0 and 10.0 for the minimum score."
            )

            try:
                min_user_score = float(min_user_score_str)
                if min_user_score < 0.0 or min_user_score > 10.0:
                    st.error("Score must be between 0.0 and 10.0.")
                    min_user_score = 0.0
            except ValueError:
                st.error("Please enter a valid numeric score.")
                min_user_score = 0.0

            # Recommendations button
            if st.button("Get Recommendations"):
                with st.spinner("Processing your request..."):
                    try:
                        recommended_games = recommend_games(df_uploaded, {'Genres': genres, 'Minimum User Score': min_user_score})
                        if not recommended_games.empty:
                            top_10_games = recommended_games.head(10)
                            st.write("### Top 10 Recommended Games")
                            st.dataframe(top_10_games)

                            # Download 
                            csv = top_10_games.to_csv(index=False)
                            st.download_button(
                                label="Download Recommendations as CSV",
                                data=csv,
                                file_name='recommended_games.csv',
                                mime='text/csv'
                            )
                        else:
                            st.warning("No games match your preferences. Try adjusting the genre or score.")
                    except Exception as e:
                        st.error(f"An error occurred while processing recommendations: {e}")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

# Page 3: Game Correlation Finder
elif page == "Game Correlation Finder":
    st.title('🎮 Game Correlation Finder')
    st.markdown("Find out how games are related based on user ratings!")

    @st.cache_data
    def load_data():
        path = 'all_video_games(cleaned).csv'
        df = pd.read_csv(path)
        path_user = 'User_Dataset.csv'
        userset = pd.read_csv(path_user)
        data = pd.merge(df, userset, on='Title').dropna()  
        return data

    data = load_data()

    score_matrix = data.pivot_table(index='user_id', columns='Title', values='user_score', fill_value=0)

    game_titles = score_matrix.columns.sort_values().tolist()

    # Split layout into two columns for better organization
    col1, col2 = st.columns([1, 3])

    with col1:
        # Game title selection
        game_title = st.selectbox("Select a game title", game_titles, help="Choose a game to see its correlation with others.")

    # Divider line for better visual separation
    st.markdown("---")

    if game_title:
        game_user_score = score_matrix[game_title]
        similar_to_game = score_matrix.corrwith(game_user_score)
        corr_drive = pd.DataFrame(similar_to_game, columns=['Correlation']).dropna()

        # Display the top 10 correlated games in the second column
        with col2:
            st.subheader(f"🎯 Correlations for '{game_title}'")
            st.dataframe(corr_drive.sort_values('Correlation', ascending=False).head(10))

        # Display number of user scores for each game
        user_scores_count = data.groupby('Title')['user_score'].count().rename('total num_of_user_score')
        merged_corr_drive = corr_drive.join(user_scores_count, how='left')

        # Calculate average user score for each game
        avg_user_score = data.groupby('Title')['user_score'].mean().rename('avg_user_score')
        detailed_corr_info = merged_corr_drive.join(avg_user_score, how='left')

        # Add developer and publisher columns (assuming they're in the dataset)
        additional_info = data[['Title', 'Developer', 'Genres']].drop_duplicates().set_index('Title')
        detailed_corr_info = detailed_corr_info.join(additional_info, how='left')

        # Show detailed high-score correlations with more information
        st.subheader("Games you may like (with more than 10 number of user scores and average user score):")
        high_score_corr = detailed_corr_info[detailed_corr_info['total num_of_user_score'] > 10].sort_values('Correlation', ascending=False).head()
        
        # Display the dataframe with additional details including average user score
        st.dataframe(high_score_corr[['Correlation', 'total num_of_user_score', 'avg_user_score', 'Developer', 'Genres']])

    else:
        st.warning("Please select a game title from the dropdown to see the correlations.")


# About Page
elif page == "About":
    st.title("📖 About")
    st.markdown("""
    This Game Recommendation System was developed to help gamers discover new titles they might enjoy.
    
    ### How It Works:
    - *Content-Based Recommendations*: This method uses the genres, platforms, and publishers of games to find similar titles.
    - *Top 10 Recommendations*: Upload your own dataset and apply filters to get personalized game recommendations.
    - *Game Correlation Finder*: Analyze game correlations based on user ratings to find titles that have similar user reception.
    
    ### Technologies Used:
    - *Python*: For building the recommendation algorithms.
    - *Streamlit*: For creating an interactive and user-friendly interface.
    - *Pandas & Scikit-learn*: For data manipulation and machine learning.

    ### Creator:
    This app was created as part of an AI project to explore recommendation systems in gaming.
    """)
    
# Footer
st.markdown("""
<div class='footer'>
    <p>Powered by Streamlit • Game Recommendation System • 2023</p>
</div>
""", unsafe_allow_html=True)
