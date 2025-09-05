import streamlit as st
import pickle
import torch
import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
import joblib

# Set page config
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #ff6b6b;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    margin: 1rem 0;
}
.genre-tag {
    display: inline-block;
    background: #4ecdc4;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin: 0.2rem;
    font-weight: bold;
}
.confidence-bar {
    background: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
    margin: 0.2rem 0;
}
.confidence-fill {
    background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
    height: 20px;
    display: flex;
    align-items: center;
    padding-left: 10px;
    color: white;
    font-weight: bold;
}
.example-box {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0 5px 5px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models and components individually (no big pickle)."""
    try:
        components = {
            'tfidf_vectorizer': joblib.load('tfidf_vectorizer.pkl'),
            'ovr_lgbm': joblib.load('lgbm_model.pkl.gz'),
            'ovr_logreg': joblib.load('logreg_model.pkl'),
            'transformer_model': SentenceTransformer('all-MiniLM-L6-v2', device='cpu'),
            'genres': [
                'action', 'adult', 'adventure', 'animation', 'biography', 'comedy',
                'crime', 'documentary', 'drama', 'family', 'fantasy', 'game-show',
                'history', 'horror', 'music', 'musical', 'mystery', 'news',
                'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show',
                'thriller', 'war', 'western'
            ]
        }
        return components
    except Exception as e:
        st.error(f"Error loading individual components: {str(e)}")
        return None


def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]*>", " ", text)  # Remove HTML tags
    text = re.sub(r"[^\w\s]", " ", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text)      # Multiple spaces to single
    return text.lower().strip()

def ensemble_predict_proba(X_features, lgbm_model, logreg_model):
    """Ensemble prediction using both models"""
    proba_lgbm = lgbm_model.predict_proba(X_features)
    proba_logreg = logreg_model.predict_proba(X_features)
    return (proba_lgbm + proba_logreg) / 2

def predict_genres(plot_text, components, threshold=0.3):
    """Predict genres for a given plot"""
    # Clean the text
    clean_plot = clean_text(plot_text)
    
    if not clean_plot:
        return [], []
    
    # TF-IDF features
    plot_tfidf = components['tfidf_vectorizer'].transform([clean_plot])
    
    # Transformer embeddings
    plot_emb = components['transformer_model'].encode([clean_plot], convert_to_numpy=True)
    
    # Combine features
    plot_features = hstack([plot_tfidf, plot_emb])
    
    # Ensemble prediction
    proba = ensemble_predict_proba(
        plot_features, 
        components['ovr_lgbm'], 
        components['ovr_logreg']
    )[0]
    
    # Get predictions above threshold
    pred_indices = np.where(proba >= threshold)[0]
    pred_genres = [components['genres'][i] for i in pred_indices]
    confidences = proba[pred_indices]
    
    # Sort by confidence
    sorted_pairs = sorted(zip(pred_genres, confidences), key=lambda x: x[1], reverse=True)
    
    return [pair[0] for pair in sorted_pairs], [pair[1] for pair in sorted_pairs]

def main():
    # Title
    st.markdown('<h1 class="main-header">🎬 Movie Genre Classifier</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models... This may take a moment on first run."):
        components = load_models()
    
    if components is None:
        st.error("Failed to load models. Please ensure all model files are present.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    st.sidebar.title("🎯 Model Settings")
    
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.05,
        help="Minimum confidence required to predict a genre"
    )
    
    show_all_probabilities = st.sidebar.checkbox(
        "Show all genre probabilities",
        help="Display confidence scores for all genres"
    )
    
    # Main interface
    st.markdown("### 📝 Enter Movie Plot Description")
    
    # Example plots
    example_plots = [
        "A young wizard attends a magical school and battles dark forces threatening the world.",
        "Two people from different backgrounds fall in love despite overwhelming obstacles.",
        "A documentary exploring the wildlife and ecosystems of the Amazon rainforest.",
        "A detective investigates a series of murders in a small industrial town.",
        "In a dystopian future, rebels fight against an oppressive government regime.",
        "A family struggles to survive during a zombie apocalypse.",
        "A group of friends embark on a treasure hunt in the Amazon jungle."
    ]
    
    selected_example = st.selectbox(
        "Or choose an example plot:",
        [""] + example_plots,
        help="Select an example to auto-fill the text area"
    )
    
    # Text input
    plot_text = st.text_area(
        "Movie Plot Description:",
        value=selected_example if selected_example else "",
        height=150,
        placeholder="Enter a detailed description of the movie plot...",
        help="Provide a detailed plot description for better genre prediction accuracy"
    )
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 Predict Genres", type="primary", use_container_width=True)
    
    if predict_button and plot_text.strip():
        with st.spinner("Analyzing plot and predicting genres..."):
            try:
                predicted_genres, confidences = predict_genres(plot_text, components, threshold)
                
                if predicted_genres:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown("### 🎭 Predicted Genres")
                    
                    # Display predicted genres with confidence
                    cols = st.columns(min(3, len(predicted_genres)))
                    for i, (genre, conf) in enumerate(zip(predicted_genres, confidences)):
                        with cols[i % 3]:
                            st.metric(
                                label=genre.title(),
                                value=f"{conf:.1%}",
                                help=f"Confidence: {conf:.3f}"
                            )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Detailed confidence breakdown
                    st.markdown("### 📊 Confidence Breakdown")
                    for genre, conf in zip(predicted_genres, confidences):
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {conf*100}%;">
                                {genre.title()}: {conf:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.warning(f"No genres predicted with confidence >= {threshold:.1%}. Try lowering the threshold.")
                
                # Show all probabilities if requested
                if show_all_probabilities:
                    st.markdown("### 📈 All Genre Probabilities")
                    
                    # Get probabilities for all genres
                    clean_plot = clean_text(plot_text)
                    plot_tfidf = components['tfidf_vectorizer'].transform([clean_plot])
                    plot_emb = components['transformer_model'].encode([clean_plot], convert_to_numpy=True)
                    plot_features = hstack([plot_tfidf, plot_emb])
                    all_proba = ensemble_predict_proba(plot_features, components['ovr_lgbm'], components['ovr_logreg'])[0]
                    
                    # Create DataFrame for better display
                    prob_df = pd.DataFrame({
                        'Genre': [g.title() for g in components['genres']],
                        'Confidence': all_proba,
                        'Percentage': [f"{p:.1%}" for p in all_proba]
                    }).sort_values('Confidence', ascending=False)
                    
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    
    elif predict_button:
        st.warning("Please enter a movie plot description.")
    
    # Footer with model info
    st.markdown("---")
    st.markdown("""
    ### 🔬 Model Information
    - **Architecture**: Ensemble of LightGBM and Logistic Regression
    - **Features**: TF-IDF (5000 features) + Sentence Transformers (384 dimensions)
    - **Training Data**: 54,214 movie plot descriptions
    - **Supported Genres**: 27 different movie genres
    """)
    
    # Additional examples
    with st.expander("💡 See More Example Plots"):
        for i, example in enumerate(example_plots, 1):
            st.markdown(f'<div class="example-box"><strong>{i}.</strong> {example}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

