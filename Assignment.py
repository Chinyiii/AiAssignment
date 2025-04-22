import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score  # NEW

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

"""
GAME RECOMMENDATION SYSTEM - CONTENT-BASED FILTERING  # NEW
Algorithm:
1. Combines Genres, Platforms, Publisher into 'content' feature
2. TF-IDF transforms text features into numerical vectors
3. Cosine similarity between game vectors
4. Returns top-N most similar games with quality metrics
"""  # NEW

pages = {
    "Home": "🏠",
    "Content-Based Recommendations": "🔍",
    "Top 10 Recommendation based on User Preferences": "📈",
    "Game Correlation Finder": "🔗",
    "About": "ℹ️"
}

# Load the data for content-based recommendations
@st.cache_data
def load_content_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df = df.dropna(subset=['Genres', 'Platforms', 'Publisher', 'User Score'])
    df['User Score'] = df['User Score'].astype(float)
    df['content'] = df['Genres'] + ' ' + df['Platforms'] + ' ' + df['Publisher']
    return df

# NEW: Evaluation function
def evaluate_recommendations(base_game, recommendations, df):
    """Calculate precision/recall for recommendations"""
    try:
        base_genres = set(df[df['Title'] == base_game]['Genres'].iloc[0].split(','))
        base_platforms = set(df[df['Title'] == base_game]['Platforms'].iloc[0].split(','))
        
        true_positives = []
        for idx, row in df.iterrows():
            game_genres = set(row['Genres'].split(','))
            game_platforms = set(row['Platforms'].split(','))
            if len(base_genres & game_genres) > 0 and len(base_platforms & game_platforms) > 0:
                true_positives.append(row['Title'])
        
        all_games = df['Title'].unique()
        y_true = [1 if game in true_positives else 0 for game in all_games]
        y_pred = [1 if game in recommendations['Title'].values else 0 for game in all_games]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        return precision, recall
    except Exception as e:
        st.error(f"Evaluation error: {str(e)}")
        return 0, 0

# Function to recommend games based on cosine similarity
def content_based_recommendations(game_name, num_recommendations=5, df=df_content):  # NEW: added df parameter
    try:
        if game_name not in df['Title'].values:  # NEW: validation
            raise ValueError("Game not found in database")
            
        if num_recommendations < 1 or num_recommendations > 20:  # NEW: validation
            raise ValueError("Number of recommendations must be 1-20")

        vectorizer = TfidfVectorizer(stop_words='english')
        content_matrix = vectorizer.fit_transform(df['content'])

        cosine_sim = cosine_similarity(content_matrix, content_matrix)
        idx = df[df['Title'].str.lower() == game_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        recommendations = df.iloc[sim_indices][['Title', 'Genres', 'User Score', 'Platforms', 'Release Date']]
        
        return recommendations
    
    except ValueError as e:  # NEW: specific error handling
        st.error(str(e))
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame()

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
    st.title("🎮 Welcome to the Game Recommendation System")

    st.markdown("""
    <div style='text-align: center;'>
        <h2 style='color: #4CAF50; font-family: "Comic Sans MS", cursive; font-size: 2.5em;'>Discover Your Next Favorite Game!</h2>
        <p style='font-size: 18px; color: white; font-family: "Arial", sans-serif;'>
            Our Game Recommendation System helps you find games you’ll love based on various methods.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Section: Content-Based Recommendations
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='color: #2196F3; font-family: "Verdana", sans-serif; font-size: 1.8em;'>🔍 Content-Based Recommendations</h3>
        <p style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6;'>
            Find games similar to the ones you already enjoy! Enter the title of your favorite game, and we'll recommend similar titles based on genres, platforms, and publishers.
        </p>
        <ul style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6; list-style-type: square;'>
            <li>Discover new titles that match your interests.</li>
            <li>Get personalized recommendations based on your game library.</li>
            <li>Easy to use with just a few clicks.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Top 10 Recommendations based on User Preferences
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='color: #4CAF50; font-family: "Verdana", sans-serif; font-size: 1.8em;'>📈 Top 10 Recommendations based on User Preferences</h3>
        <p style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6;'>
            Upload your own game data and filter results based on your preferences. Customize your search by entering preferred genres and minimum user scores to get the top 10 recommendations tailored just for you.
        </p>
        <ul style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6; list-style-type: square;'>
            <li>Upload your latest dataset for up-to-date recommendations.</li>
            <li>Apply filters to match your taste and preferences.</li>
            <li>Download your personalized recommendations as a CSV file.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Game Correlation Finder
    st.markdown("""
    <div style='background-color: #fff3e0; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='color: #FF5722; font-family: "Verdana", sans-serif; font-size: 1.8em;'>🔗 Game Correlation Finder</h3>
        <p style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6;'>
            Explore how games are related based on user ratings. Select a game to see its correlation with others, and find out which games share similar user reception.
        </p>
        <ul style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6; list-style-type: square;'>
            <li>Identify games with similar user scores.</li>
            <li>Understand game relationships through detailed correlations.</li>
            <li>Discover new games based on user ratings and correlations.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 40px; font-family: "Arial", sans-serif;'>
        <h4>Use the navigation sidebar to explore different features of the app.</h4>
    </div>
    """, unsafe_allow_html=True)

# Page 1: Content-Based Recommendations
elif page == "Content-Based Recommendations":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Content-Based Game Recommendations</h1>", unsafe_allow_html=True)  # NEW: updated title
    
    with st.expander("How this works"):  # NEW: explanation
        st.markdown("""
        This system recommends games based on:
        - 🎭 Genres
        - 🖥️ Platforms
        - 🏢 Publisher
        using TF-IDF and cosine similarity algorithms.
        """)
    
    game_list = df_content['Title'].unique()
    game_input = st.selectbox("Choose a game from the list:", game_list, 
                            help="Select a game to find similar titles")  # NEW: help text
    
    num_recommendations = st.slider('Number of recommendations', 1, 10, 5,
                                  help="Choose how many recommendations to display")  # NEW: help text

    if game_input:
        with st.container(border=True):  # NEW: better game info display
            st.markdown(f"### 🕹️ Selected Game: {game_input}")
            game_info = df_content[df_content['Title'] == game_input].iloc[0]
            cols = st.columns(4)
            cols[0].write(f"**Genres**: {game_info['Genres']}")
            cols[1].write(f"**Platforms**: {game_info['Platforms']}")
            cols[2].write(f"**Publisher**: {game_info['Publisher']}")
            cols[3].write(f"**User Score**: ⭐ {game_info['User Score']}/10")

    if st.button('Get Recommendations', type="primary"):  # NEW: primary button
        with st.spinner('Finding similar games...'):  # NEW: loading indicator
            recommendations = content_based_recommendations(game_input, num_recommendations)
            
            if not recommendations.empty:
                st.success(f"Found {len(recommendations)} similar games!")  # NEW: success message
                
                # NEW: Evaluation metrics display
                precision, recall = evaluate_recommendations(game_input, recommendations, df_content)
                st.metric(label="Recommendation Quality", 
                         value=f"Precision: {precision:.1%} | Recall: {recall:.1%}",
                         help="Higher values mean better recommendations")
                
                # NEW: Card-based display
                st.subheader("🎯 Recommended Games")
                cols = st.columns(3)
                for idx, row in recommendations.iterrows():
                    with cols[idx % 3]:
                        with st.container(border=True):
                            st.markdown(f"**{row['Title']}**")
                            st.write(f"⭐ {row['User Score']}/10")
                            st.caption(f"**Genres**: {row['Genres']}")
                            st.caption(f"**Platforms**: {row['Platforms']}")
                            st.caption(f"Released: {row['Release Date']}")
                
                # NEW: Debug info for presentation
                if st.checkbox("Show technical details"):
                    st.write("Similarity calculation details:")
                    st.code("""
                    # Algorithm Steps:
                    1. Combine features (Genres + Platforms + Publisher)
                    2. Convert to TF-IDF vectors
                    3. Compute cosine similarity matrix
                    4. Return most similar games
                    """)
            else:
                st.warning("No recommendations found. Try a different game.")

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
st.markdown("<h5 style='text-align: center;'>Powered by Streamlit</h5>", unsafe_allow_html=True)
