# 📧 SMS Spam Classification using Machine Learning  

A complete **end-to-end machine learning project** to classify SMS messages as **Spam 🚫** or **Ham (Not Spam) ✅**, featuring:  
- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Text vectorization using **TF-IDF**  
- Model training & evaluation (Naive Bayes, Logistic Regression, SVM, Random Forest)  
- Model deployment with **Streamlit App** and **prediction history** tracking  

---

## 📑 Table of Contents  
1. [Project Overview](#project-overview)  
2. [Dataset & Sources](#dataset--sources)  
3. [Setup (Requirements)](#setup-requirements)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
5. [Model Training & Evaluation](#model-training--evaluation)  
6. [Model Comparison](#model-comparison)  
7. [Streamlit App](#streamlit-app)  
8. [Results](#results)  
9. [Conclusion & Future Work](#conclusion--future-work)  
10. [About the Author](#about-the-author)  

---

## 📌 Project Overview  

The goal of this project is to build an **SMS Spam Classifier** that can distinguish between spam messages and legitimate messages (ham).  

### Steps involved:  
- Preprocessing raw SMS messages (cleaning, removing stopwords, lemmatization)  
- Feature extraction using **TF-IDF Vectorizer**  
- Training multiple ML models for classification  
- Selecting the **best performing model**  
- Deploying the model in a **Streamlit web app**  

---

## 📊 Dataset & Sources  

The dataset used is the **SMS Spam Collection Dataset**, containing **5,574 SMS messages** labeled as:  

- `ham` → legitimate message  
- `spam` → unsolicited message  

📄 Source & Citation: [SMS Spam Collection](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)  

---

## ⚙️ Setup (Requirements)  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/JaveriaJameel/sms-spam-classifier.git
cd sms-spam-classifier
pip install -r requirements.txt
requirements.txt
pandas
numpy
matplotlib
seaborn
wordcloud
scikit-learn
nltk
joblib
streamlit
📊 Exploratory Data Analysis (EDA)

- Class distribution: Dataset is slightly imbalanced (ham > spam).

- Message length analysis: Spam messages tend to be longer.

- WordClouds: Visualized most frequent words in ham vs spam.

- N-grams (bigrams/trigrams): Extracted frequent word sequences.

- Correlation heatmap: Explored relationships between engineered features.

🤖 Model Training & Evaluation

- Trained multiple models on TF-IDF features:

- Naive Bayes (fast & baseline model)

- Logistic Regression (best overall performance)

- SVM (strong performance, slower)

- Random Forest (powerful, risk of overfitting)

- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
📊 Model Comparison
| Model               | Accuracy | F1-score |
| ------------------- | -------- | -------- |
| Naive Bayes         | \~0.97   | \~0.95   |
| Logistic Regression | \~0.98   | \~0.96   |
| SVM                 | \~0.97   | \~0.95   |
| Random Forest       | \~0.97   | \~0.94   |
✅ Final Choice: Logistic Regression
---
🌐 Streamlit App

The project includes a Streamlit Web App for real-time predictions.

Features:

- Enter any SMS text → Get classification (Spam/Ham)

- Prediction history stored in CSV file

- Search & filter past predictions

- Download full prediction history

- Clear history with one click

Run the app locally:
streamlit run app.py

📊 Results

- Achieved ~98% accuracy using Logistic Regression.

- Strong precision & recall for spam detection.

- Deployed as an interactive Streamlit App.
---
🚀 Conclusion & Future Work

- Successfully built a robust SMS Spam Classifier.

- Logistic Regression emerged as the best model.

- Streamlit app enhances usability with history tracking.

🔮 Future improvements:

- Use deep learning models (LSTMs, BERT) for better performance.

- Expand dataset to multi-language SMS.

- Deploy on cloud platforms (Heroku, AWS, etc.).
---
👩‍💻 About the Author

Javeria Jameel 
Data Scientist (Python • SQL • ML & Analytics) based in Pakistan

I build practical, portfolio-ready projects that turn raw data into clear insights and deployable models.

Focus Areas:

- 📝 NLP (Text Classification)

- 🎯 Recommender Systems

- 📊 Predictive Modeling

📫 Connect with me:

GitHub: https://github.com/JaveriaJameel

LinkedIn: https://www.linkedin.com/in/javeria-jameel-88a95a380/
⭐ If you find this project helpful, don’t forget to star the repo!


---

👉 Do you also want me to create a **`requirements.txt`** file automatically from your Jupyter + Streamlit code so you can directly push both files to GitHub?


