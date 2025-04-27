import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        
def calculate_mae(recommendations, target_game):
    try:
        # Features to compare (numeric features only)
        numeric_features = ['User Score']
        
        # Calculate MAE for each feature
        mae_results = {}
        for feature in numeric_features:
            target_value = target_game[feature]
            recommended_values = recommendations[feature]
            absolute_errors = abs(recommended_values - target_value)
            mae_results[f"{feature} MAE"] = absolute_errors.mean()

        # Combine target and recommendations genres into a list
        all_genres = [target_game['Genres']] + recommendations['Genres'].tolist()
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_genres)
        
        # Calculate cosine similarities (first row is target)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Cosine similarity is 1 for perfect match, so error = 1 - cosine_sim
        genre_errors = 1 - cosine_similarities
        
        # Mean of genre errors
        mae_results["Genre Similarity Error"] = genre_errors.mean()
        
        return mae_results
    
    except Exception as e:
        st.error(f"Could not calculate MAE: {str(e)}")
        return None
        
# Function to recommend games based on file upload and filters
def recommend_games(df, preferences):
    genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
    score_filter = df['User Score'] >= preferences['Minimum User Score']
    filtered_df = df[genre_filter & score_filter]
    return filtered_df

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


#Page 1 : Content Based Filtering
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
            
            st.markdown("---")
            st.subheader("Recommendation Accuracy Metrics")
            
            # Calculate MAE
            mae_results =calculate_mae(recommendations, game_info)
            
            if mae_results:
                # Display MAE metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("User Score MAE", 
                             f"{mae_results.get('User Score MAE', 0):.2f}",
                             help="Lower is better (0 = perfect match)")
                
                with col2:
                    st.metric("Genre Similarity Error", 
                             f"{mae_results.get('Genre Similarity Error', 0):.2f}",
                             help="Lower is better (0 = perfect match)")
                
                with col3:
                    avg_mae = (mae_results.get('User Score MAE', 0) + 
                              mae_results.get('Genre Similarity Error', 0)) / 2
                    st.metric("Overall MAE", 
                             f"{avg_mae:.2f}",
                             help="Average of all feature errors")
                
                # Visualize the errors
                with st.expander("View Detailed Error Analysis"):
                    # Create a DataFrame for visualization
                    mae_df = pd.DataFrame({
                        'Metric': ['User Score', 'Genre Similarity'],
                        'MAE': [mae_results['User Score MAE'], 
                               mae_results['Genre Similarity Error']]
                    })
                    
                    # Plot using Streamlit's native bar chart
                    st.bar_chart(mae_df.set_index('Metric'))
                    
                   # Interpretation
                    st.markdown("""
                    **How to interpret MAE:**
                    - **User Score MAE**: Average difference in user ratings between recommended and target game
                    - **Genre Similarity Error**: 1 - cosine similarity of genres (0 = identical genres)
                    - **Lower values** indicate better matches
                    """)
            
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

    # Create pivot table
    score_matrix = data.pivot_table(index='user_id', columns='Title', values='user_score')
    game_titles = score_matrix.columns.sort_values().tolist()

    col1, col2 = st.columns([1, 3])

    with col1:
        game_title = st.selectbox("Select a game title", game_titles, help="Choose a game to see its correlation with others.")

    st.markdown("---")

    if game_title:

        # Normalize the score_matrix by subtracting user mean
        normalized_matrix = score_matrix.sub(score_matrix.mean(axis=1), axis=0)

        # Pearson Correlation (normalized)
        def pearson_correlation(matrix, target_col):
            target = matrix[target_col]
            correlations = {}
            for col in matrix.columns:
                if col == target_col:
                    continue
                common = matrix[[target_col, col]].dropna()
                if len(common) < 3:
                    correlations[col] = 0
                    continue
                corr = common[target_col].corr(common[col])
                correlations[col] = corr if not np.isnan(corr) else 0
            return pd.Series(correlations)

        pearson_corr = pearson_correlation(normalized_matrix, game_title)
        pearson_corr = pearson_corr.clip(-1, 1)  # Clip to [-1, 1]

        # Cosine Similarity (normalized)
        def cosine_similarity_optimized(matrix, target_col):
            similarities = {}
            target_vec = matrix[target_col].values

            for col in matrix.columns:
                if col == target_col:
                    continue

                other_vec = matrix[col].values
                mask = ~np.isnan(target_vec) & ~np.isnan(other_vec)
                valid_target = target_vec[mask]
                valid_other = other_vec[mask]

                if len(valid_target) < 3:
                    similarities[col] = 0
                    continue

                dot_product = np.dot(valid_target, valid_other)
                norm_product = np.linalg.norm(valid_target) * np.linalg.norm(valid_other)

                similarities[col] = dot_product / norm_product if norm_product != 0 else 0

            return pd.Series(similarities)

        cosine_sim = cosine_similarity_optimized(normalized_matrix, game_title)
        cosine_sim = cosine_sim.clip(0, 1)  # Clip to [0, 1]

        # Hybrid Score
        hybrid_score = 0.6 * pearson_corr + 0.4 * cosine_sim
        hybrid_df = pd.DataFrame({'Hybrid Score': hybrid_score})

        # Metadata
        meta_info = data[['Title', 'Developer', 'Genres']].drop_duplicates().set_index('Title')
        user_counts = data.groupby('Title')['user_score'].count().rename('User Ratings Count')
        avg_scores = data.groupby('Title')['user_score'].mean().rename('Average Score')

        # Shared users calculation
        shared_users = {}
        target_users = set(score_matrix[score_matrix[game_title].notna()].index)
        for game in hybrid_df.index:
            game_users = set(score_matrix[score_matrix[game].notna()].index)
            shared_users[game] = len(target_users & game_users)
        shared_users = pd.Series(shared_users, name='Shared Users')

        detailed = hybrid_df.join([
            shared_users,
            user_counts,
            avg_scores,
            meta_info
        ], how='left')

        # Apply filtering thresholds
        detailed = detailed[
            (detailed['Shared Users'] >= 3) &
            (detailed['User Ratings Count'] >= 5) &
            (detailed['Average Score'] >= 6.0)
        ].copy()

        # Compute Accuracy
        def compute_metrics(game1, game2, threshold=1.0):
            common = score_matrix[[game1, game2]].dropna()
            if common.empty:
                return None, None
            diffs = (common[game1] - common[game2]).abs()
            tp = (diffs <= threshold).sum()
            total = len(diffs)
            accuracy = round((tp / total) * 100, 2) if total > 0 else 0
           
            return accuracy

        if not detailed.empty:
            # Accuracy 
            metrics = [compute_metrics(game_title, game) for game in detailed.index]
            accuracy_df = pd.DataFrame(
                metrics,
                index=detailed.index,
                columns=['Accuracy (%)']
            )
            detailed = pd.concat([detailed, accuracy_df], axis=1)

            # Calculate MAE for top 10
            top10_recommended = detailed.nlargest(10, 'Hybrid Score').index
            maes = []
            for game in top10_recommended:
                common = score_matrix[[game_title, game]].dropna()
                if not common.empty:
                    mae = mean_absolute_error(common[game_title], common[game])
                    maes.append(mae)
            avg_mae = round(np.mean(maes), 4) if maes else None

            # Display
            with col2:
                st.subheader(f"🎯 Enhanced Correlation (Normalized Pearson + Cosine) for '{game_title}'")
                st.dataframe(
                    detailed[['Hybrid Score', 'Accuracy (%)', 'Shared Users', 
                              'User Ratings Count', 'Average Score', 'Developer', 'Genres']]
                    .sort_values('Hybrid Score', ascending=False)
                    .head(10)
                )

                st.markdown("---")
                st.subheader("📈 Recommendation Quality Metrics")
                if avg_mae is not None:
                    st.success(f"*Average MAE for Top 10 Recommendations:* {avg_mae}")

                else:
                    st.warning("Not enough common users to calculate MAE.")
        else:
            st.warning("No sufficient data to show related games for this title. Try another more popular game.")

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
