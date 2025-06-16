import streamlit as st
import joblib
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go


# Preprocess text function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join(text.split())
    return text


# Load vectorizer, label encoder, and model results
try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    model_results = joblib.load("model_results.pkl")
except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e}. Please run the training script to generate all necessary files."
    )
    st.stop()

# Model file paths
model_files = {
    "Naive Bayes": "naive_bayes_model.pkl",
    "SVM": "svm_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
}


# Cached model loading function
@st.cache_resource
def load_model(model_name):
    try:
        return joblib.load(model_files[model_name])
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        st.stop()


# Streamlit app layout
st.title("📊📰 Newslyzer - Classify Articles")
st.markdown(
    """
Predict the category of a news article using machine learning models trained on BBC news articles.
"""
)

# Mode selection
mode = st.selectbox("Select mode:", ["Single Model", "Compare All Models"])

# Model selection for Single Model mode
if mode == "Single Model":
    selected_model_name = st.selectbox("Select a model:", list(model_files.keys()))
    accuracy = model_results[selected_model_name]["accuracy"]
    st.markdown(f"**Selected Model:** {selected_model_name} (Accuracy: {accuracy:.2%})")

# Display possible categories
categories = list(label_encoder.classes_)
st.markdown("**Possible categories:** " + ", ".join(categories))

# Instructions and sample text
st.markdown(
    """
### Instructions
Enter a news article text below or select a sample to test the model(s).
"""
)
sample_texts = {
    "Tech Sample": "The new AI model from OpenAI is absolutely blowing minds with its ability to generate photorealistic images from just a few words. It's raising some serious questions about what's real and what's not, but the creative possibilities are insane.",
    "Sport Sample": "What a game! The Hyderabad Hunters just clinched the national cricket championship in a heart-stopping super over. The crowd went wild – definitely a match for the history books, and a massive win for the city!",
    "Politics Sample": "After weeks of intense debate, Parliament finally passed the new healthcare reform bill last night. It's a landmark moment, though critics are already pointing out potential challenges in implementation across different states.",
    "Business Sample": "Startup funding seems to be cooling down a bit, but this week saw a massive acquisition: a little-known sustainable energy firm was bought out by a global conglomerate for billions. It signals a big shift towards green investments.",
    "Entertainment Sample": "Everyone's talking about the new Telugu blockbuster that just hit theaters. It's not just breaking box office records, but the performances are getting rave reviews, and people are calling it a game-changer for regional cinema.",
}
selected_sample = st.selectbox(
    "Select a sample text (optional):", ["None"] + list(sample_texts.keys())
)
default_text = sample_texts[selected_sample] if selected_sample != "None" else ""

# Text input
user_input = st.text_area(
    "Enter your news article text here:", value=default_text, height=200
)

# Prediction
if st.button("Predict"):
    if not user_input.strip():
        st.error("Please enter some text to predict.")
    else:
        with st.spinner("Predicting..."):
            processed_text = preprocess_text(user_input)
            text_tfidf = vectorizer.transform([processed_text])

            if mode == "Single Model":
                model = load_model(selected_model_name)
                prediction = model.predict(text_tfidf)
                category = label_encoder.inverse_transform(prediction)[0]
                st.success(f"Predicted Category: **{category}**")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(text_tfidf)[0]
                    prob_df = pd.DataFrame(
                        {"Category": categories, "Probability": probabilities}
                    )
                    prob_df = prob_df.sort_values("Probability", ascending=False)
                    st.markdown("**Prediction Probabilities:**")
                    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

                    # Interactive vertical bar graph
                    st.markdown("**Confidence Across Categories**")
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=categories,
                                y=probabilities,
                                text=[f"{p:.2%}" for p in probabilities],
                                textposition="auto",
                                marker_color="rgb(99, 110, 250)",
                            )
                        ]
                    )
                    fig.update_layout(
                        title=f"Probability Distribution - {selected_model_name}",
                        xaxis_title="Category",
                        yaxis_title="Probability",
                        yaxis_range=[0, 1],
                        showlegend=False,
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        "Selected model does not support probability predictions or confidence visualization."
                    )

            elif mode == "Compare All Models":
                results = []
                prob_data = {model_name: None for model_name in model_files.keys()}
                for model_name in model_files.keys():
                    model = load_model(model_name)
                    prediction = model.predict(text_tfidf)
                    category = label_encoder.inverse_transform(prediction)[0]
                    accuracy = model_results[model_name]["accuracy"]
                    confidence_str = "N/A"
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(text_tfidf)[0]
                        confidence = max(probabilities)
                        confidence_str = f"{confidence:.2%}"
                        prob_data[model_name] = probabilities
                    results.append(
                        {
                            "Model": model_name,
                            "Prediction": category,
                            "Accuracy": f"{accuracy:.2%}",
                            "Confidence": confidence_str,
                        }
                    )
                df = pd.DataFrame(results)
                st.markdown("**Predictions from All Models:**")
                st.table(df)

                # Interactive vertical bar graphs for all models
                st.markdown("**Confidence Across Categories for All Models**")
                for model_name in model_files.keys():
                    if prob_data[model_name] is not None:
                        prob_df = pd.DataFrame(
                            {
                                "Category": categories,
                                "Probability": prob_data[model_name],
                            }
                        )
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    x=categories,
                                    y=prob_data[model_name],
                                    text=[f"{p:.2%}" for p in prob_data[model_name]],
                                    textposition="auto",
                                    marker_color="rgb(99, 110, 250)",
                                )
                            ]
                        )
                        fig.update_layout(
                            title=f"Probability Distribution - {model_name}",
                            xaxis_title="Category",
                            yaxis_title="Probability",
                            yaxis_range=[0, 1],
                            showlegend=False,
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(
                            f"{model_name} does not support probability predictions or confidence visualization."
                        )

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    """
This app allows you to classify BBC news articles into categories such as *business, entertainment, politics, sport, tech* using different machine learning models. Choose 'Single Model' to use one model with an interactive probability visualization, or 'Compare All Models' to see predictions and confidence distributions from all models.
"""
)
