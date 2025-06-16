import streamlit as st
import joblib
import pandas as pd
import re
import plotly.express as px

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

# ---- Inject CSS only for prediction cards, not tabs ----
def inject_card_css():
    css = """
    <style>
    /* Prediction cards styling */
    .prediction-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .prediction-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    .prediction-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.95;
    }
    .prediction-category {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
        gap: 0.5rem;
    }
    .metric-item {
        text-align: center;
        padding: 0.8rem;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        flex: 1;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    /* Category-specific gradients */
    .prediction-tech {
        background: linear-gradient(135deg, #3742fa 0%, #2f3542 100%);
    }
    .prediction-sport {
        background: linear-gradient(135deg, #2ed573 0%, #1e90ff 100%);
    }
    .prediction-business {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
    }
    .prediction-politics {
        background: linear-gradient(135deg, #5f27cd 0%, #341f97 100%);
    }
    .prediction-entertainment {
        background: linear-gradient(135deg, #ff3838 0%, #ff9ff3 100%);
    }
    .single-prediction {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Inject card CSS at start
inject_card_css()

# Streamlit app layout
st.title("📊📰 Newslyzer - Classify Articles")
st.markdown(
    """
    Predict the category of a news article using machine learning models trained on BBC news articles.
    """
)

# Display possible categories
categories = list(label_encoder.classes_)
st.markdown(
    "**Available categories:** " + " • ".join([f"*{cat}*" for cat in categories])
)

# Model selection
st.markdown("### 🤖 Model Selection")
selected_models = st.multiselect(
    "Choose one or more models to compare:",
    list(model_files.keys()),
    default=["Naive Bayes"],
    help="Select machine learning models for prediction",
)

# Sample text selection
st.markdown("### 📝 Sample Articles")
sample_texts = {
    "Tech Sample": "The new AI model from OpenAI is absolutely blowing minds with its ability to generate photorealistic images from just a few words. It's raising some serious questions about what's real and what's not, but the creative possibilities are insane.",
    "Sport Sample": "What a game! The Hyderabad Hunters just clinched the national cricket championship in a heart-stopping super over. The crowd went wild – definitely a match for the history books, and a massive win for the city!",
    "Politics Sample": "After weeks of intense debate, Parliament finally passed the new healthcare reform bill last night. It's a landmark moment, though critics are already pointing out potential challenges in implementation across different states.",
    "Business Sample": "Startup funding seems to be cooling down a bit, but this week saw a massive acquisition: a little-known sustainable energy firm was bought out by a global conglomerate for billions. It signals a big shift towards green investments.",
    "Entertainment Sample": "Everyone's talking about the new Telugu blockbuster that just hit theaters. It's not just breaking box office records, but the performances are getting rave reviews, and people are calling it a game-changer for regional cinema.",
}
selected_sample = st.selectbox(
    "Try a sample article (optional):",
    ["None"] + list(sample_texts.keys()),
    help="Select a pre-written sample to test the model",
)
default_text = sample_texts[selected_sample] if selected_sample != "None" else ""

# Form for text input with enter key submission
st.markdown("### ✍️ Article Input")
with st.form("article_form", clear_on_submit=False):
    user_input = st.text_area(
        "Enter your news article text here:",
        value=default_text,
        height=200,
        help="Paste or type your news article content here. Press Ctrl+Enter to submit.",
    )
    submitted = st.form_submit_button(
        "🔍 Analyze Article",
        use_container_width=True,
        help="Click to analyze or press Enter",
    )

# Prediction logic
if submitted:
    if not selected_models:
        st.error("⚠️ Please select at least one model to analyze.")
    elif not user_input.strip():
        st.error("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔄 Analyzing article..."):
            processed_text = preprocess_text(user_input)
            text_tfidf = vectorizer.transform([processed_text])

            results = []
            prob_data = {model_name: None for model_name in selected_models}

            for model_name in selected_models:
                model = load_model(model_name)
                try:
                    prediction = model.predict(text_tfidf)
                except Exception as e:
                    st.error(f"Error during prediction with {model_name}: {e}")
                    continue
                if len(prediction) == 0:
                    st.error(f"No prediction returned by {model_name}.")
                    continue
                category = label_encoder.inverse_transform(prediction)[0]
                accuracy = model_results.get(model_name, {}).get("accuracy", None)
                accuracy_str = f"{accuracy:.2%}" if accuracy is not None else "N/A"
                confidence_str = "N/A"

                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(text_tfidf)[0]
                        confidence = max(probabilities)
                        confidence_str = f"{confidence:.2%}"
                        prob_data[model_name] = probabilities
                    except Exception:
                        prob_data[model_name] = None

                results.append(
                    {
                        "Model": model_name,
                        "Prediction": category,
                        "Accuracy": accuracy_str,
                        "Confidence": confidence_str,
                    }
                )

        # Enhanced comparison display
        st.markdown("### 🔍 Model Prediction Results")
        num_models = len(results)

        # CSS class mapping for prediction categories
        category_classes = {
            "tech": "prediction-tech",
            "sport": "prediction-sport",
            "business": "prediction-business",
            "politics": "prediction-politics",
            "entertainment": "prediction-entertainment",
        }

        if num_models == 0:
            st.info("No successful model predictions to display.")
        elif num_models == 1:
            result = results[0]
            category_class = category_classes.get(
                result["Prediction"].lower(), "single-prediction"
            )
            st.markdown(
                f"""
            <div class="prediction-card {category_class}">
                <div class="prediction-title">{result['Model']} Prediction</div>
                <div class="prediction-category">{result['Prediction']}</div>
                <div class="metric-container">
                    <div class="metric-item">
                        <strong>Confidence</strong><br>
                        <span style="font-size: 1.5em;">{result['Confidence']}</span>
                    </div>
                    <div class="metric-item">
                        <strong>Model Accuracy</strong><br>
                        <span style="font-size: 1.5em;">{result['Accuracy']}</span>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            # Multiple models layout
            if num_models == 2:
                cols = st.columns(2)
                for i, result in enumerate(results):
                    with cols[i]:
                        category_class = category_classes.get(
                            result["Prediction"].lower(), ""
                        )
                        st.markdown(
                            f"""
                        <div class="prediction-card {category_class}">
                            <div class="prediction-title">{result['Model']} Prediction</div>
                            <div class="prediction-category">{result['Prediction']}</div>
                            <div class="metric-container">
                                <div class="metric-item">
                                    <strong>Confidence</strong><br>
                                    <span style="font-size: 1.2em;">{result['Confidence']}</span>
                                </div>
                                <div class="metric-item">
                                    <strong>Model Accuracy</strong><br>
                                    <span style="font-size: 1.2em;">{result['Accuracy']}</span>
                                </div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
            elif num_models == 3:
                # First model full width
                result = results[0]
                category_class = category_classes.get(
                    result["Prediction"].lower(), "single-prediction"
                )
                st.markdown(
                    f"""
                <div class="prediction-card {category_class}">
                    <div class="prediction-title">{result['Model']} Prediction</div>
                    <div class="prediction-category">{result['Prediction']}</div>
                    <div class="metric-container">
                        <div class="metric-item">
                            <strong>Confidence</strong><br>
                            <span style="font-size: 1.5em;">{result['Confidence']}</span>
                        </div>
                        <div class="metric-item">
                            <strong>Model Accuracy</strong><br>
                            <span style="font-size: 1.5em;">{result['Accuracy']}</span>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                # Next two half-width
                cols = st.columns(2)
                for i, result in enumerate(results[1:3]):
                    with cols[i]:
                        category_class = category_classes.get(
                            result["Prediction"].lower(), ""
                        )
                        st.markdown(
                            f"""
                        <div class="prediction-card {category_class}">
                            <div class="prediction-title">{result['Model']} Prediction</div>
                            <div class="prediction-category">{result['Prediction']}</div>
                            <div class="metric-container">
                                <div class="metric-item">
                                    <strong>Confidence</strong><br>
                                    <span style="font-size: 1.2em;">{result['Confidence']}</span>
                                </div>
                                <div class="metric-item">
                                    <strong>Model Accuracy</strong><br>
                                    <span style="font-size: 1.2em;">{result['Accuracy']}</span>
                                </div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
            else:
                # 4 or more: display in rows of 2 columns
                for row_start in range(0, num_models, 2):
                    cols = st.columns(2)
                    for idx in range(2):
                        i = row_start + idx
                        if i >= num_models:
                            break
                        result = results[i]
                        category_class = category_classes.get(
                            result["Prediction"].lower(), ""
                        )
                        with cols[idx]:
                            st.markdown(
                                f"""
                            <div class="prediction-card {category_class}">
                                <div class="prediction-title">{result['Model']} Prediction</div>
                                <div class="prediction-category">{result['Prediction']}</div>
                                <div class="metric-container">
                                    <div class="metric-item">
                                        <strong>Confidence</strong><br>
                                        <span style="font-size: 1.2em;">{result['Confidence']}</span>
                                    </div>
                                    <div class="metric-item">
                                        <strong>Model Accuracy</strong><br>
                                        <span style="font-size: 1.2em;">{result['Accuracy']}</span>
                                    </div>
                                </div>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

        # Probability distribution charts in default-styled tabs
        st.markdown("### 📊 Probability Distribution by Model")
        if not selected_models:
            st.info("No models selected for probability charts.")
        else:
            tabs = st.tabs(selected_models)
            # Determine fallback template for Plotly based on theme base
            try:
                base = st.get_option("theme.base")
            except:
                base = "light"
            template = "plotly_dark" if base == "dark" else "plotly_white"

            for tab, model_name in zip(tabs, selected_models):
                with tab:
                    probs = prob_data.get(model_name)
                    if probs is None:
                        st.info(f"No probability data for {model_name}.")
                        continue
                    categories_all = list(label_encoder.classes_)
                    if len(probs) != len(categories_all):
                        st.error(f"Got {len(probs)} probabilities but {len(categories_all)} classes.")
                        continue
                    df_probs = pd.DataFrame({
                        "category": categories_all,
                        "probability": probs
                    }).sort_values("probability", ascending=True)
                    fig = px.bar(
                        df_probs,
                        x="probability",
                        y="category",
                        orientation="h",
                        title=f"{model_name} - Predicted Probabilities",
                        labels={"probability": "Probability", "category": "Category"},
                    )
                    fig.update_layout(
                        margin=dict(l=80, r=20, t=40, b=20),
                        template=template
                    )
                    fig.update_traces(
                        text=[f"{p:.1%}" for p in df_probs["probability"]],
                        textposition="outside",
                        cliponaxis=False
                    )
                    try:
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                    except TypeError:
                        st.plotly_chart(fig, use_container_width=True)

# Enhanced sidebar
st.sidebar.header("About Newslyzer")
st.sidebar.markdown(
    """
**Newslyzer** uses machine learning to classify BBC news articles into categories:

🏢 **Business** - Financial and economic news  
🎭 **Entertainment** - Movies, TV, celebrity news  
🏛️ **Politics** - Government and political affairs  
⚽ **Sport** - Sports news and events  
💻 **Tech** - Technology and science news

### 🔧 Models Available:
- **Naive Bayes**: Fast, probabilistic classifier
- **SVM**: Support Vector Machine (no probabilities)
- **Random Forest**: Ensemble method with confidence
- **Logistic Regression**: Statistical linear model

### 💡 Tips:
- Select multiple models to compare predictions
- Use **Enter key** to submit text quickly
- Try sample articles to see model performance
- Models with probabilities show detailed confidence breakdowns
"""
)
st.sidebar.markdown("---")
