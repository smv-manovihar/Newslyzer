# 📰 Newslyzer

**Newslyzer** is a machine learning–powered news classification project that automatically categorizes BBC news articles into predefined categories. It leverages multiple machine learning algorithms—including Logistic Regression, Naive Bayes, Random Forest, and SVM—trained on the BBC News dataset.

A **Streamlit web application** is also included for interactive exploration, live predictions, and model comparisons.

---

## ✨ Features

- ✅ **Multiple Classification Models**: Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machines
- ✅ **Text Preprocessing & Feature Extraction**: Tokenization and TF-IDF vectorization
- ✅ **Label Encoding**: Maps categories into machine-readable form
- ✅ **Model Training & Evaluation**: Performance comparison across algorithms
- ✅ **Interactive Web App**: Built with Streamlit for real-time predictions

---

## 📂 Dataset

The project uses the **BBC News Train.csv** dataset, which contains news articles along with their corresponding categories (e.g., politics, sport, tech, etc.).

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/smv-manovihar/Newslyzer.git
   cd newslyzer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to **[http://localhost:8501](http://localhost:8501)** to start using Newslyzer.

---

## 🌐 Live Demo

👉 Try the live version here: [newslyzer.streamlit.app](https://newslyzer.streamlit.app/)

---

## 📁 Project Structure

```
├── app.py                # Streamlit web application
├── ModelTrain.ipynb      # Jupyter Notebook for model training & evaluation
├── BBC News Train.csv    # Dataset
├── *.pkl                 # Trained models & encoders (serialized)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Technologies Used

- Python (scikit-learn, pandas, numpy, matplotlib)
- Natural Language Processing (TF-IDF, label encoding)
- Machine Learning Models (Logistic Regression, Naive Bayes, Random Forest, SVM)
- Streamlit (interactive web interface)

---

## 📜 License

This project is intended **for educational purposes only**.
