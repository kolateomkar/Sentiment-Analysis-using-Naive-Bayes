This project performs **Sentiment Analysis** on the **IMDB Movie Reviews dataset**.  
The goal is to classify each review as **Positive** or **Negative** based on the text content.  
It uses traditional **Machine Learning (Naive Bayes classifiers)** and manual text preprocessing techniques.

# 🎭 Sentiment Analysis using Naive Bayes

## 📘 Project Overview  
This project performs **Sentiment Analysis** on the **IMDB Movie Reviews dataset**.  
The goal is to classify each review as **Positive** or **Negative** based on the text content.  
It uses traditional **Machine Learning (Naive Bayes classifiers)** and manual text preprocessing techniques.

---

## 🧩 Features  
- Cleans raw review text by removing HTML tags, special characters, and stopwords.  
- Converts text into lowercase and applies stemming for normalization.  
- Uses **Bag of Words (CountVectorizer)** for feature extraction.  
- Trains and evaluates multiple Naive Bayes models:  
  - GaussianNB  
  - MultinomialNB  
  - BernoulliNB  
- Compares models using **Accuracy Score** and **AUC (Area Under Curve)**.  
- Saves the best model (**BernoulliNB**) using **pickle** for future predictions.

---

## 🧠 Machine Learning Workflow  
1. **Data Loading** – IMDB Dataset (CSV file)  
2. **Data Preprocessing** – Cleaning and transforming text data  
3. **Feature Extraction** – Convert text into numerical form using CountVectorizer  
4. **Model Training** – Train Naive Bayes models on the processed dataset  
5. **Evaluation** – Compare model performance using Accuracy and AUC  
6. **Prediction** – Predict sentiment for new review inputs  

---

## 🛠️ Technologies Used  
- **Python**  
- **Pandas**, **NumPy**  
- **NLTK** (for tokenization, stopword removal, stemming)  
- **Scikit-learn** (for feature extraction, model building, and evaluation)  
- **Matplotlib**, **Seaborn** (for data visualization)  
- **Pickle** (for saving models)

---

## 📂 Dataset  
**Dataset:** `IMDB Dataset.csv`  
- **Columns:**  
  - `review`: Movie review text  
  - `sentiment`: Sentiment label (`positive` / `negative`)

---

## 🧩 Future Improvements  
- Replace **CountVectorizer** with **TF-IDF** for better text representation.  
- Add **Logistic Regression** or **SVM** models for comparison.  
- Build a **Streamlit web app** for interactive sentiment testing.  
- Use **Deep Learning models (LSTM / BERT)** for advanced NLP.

---

