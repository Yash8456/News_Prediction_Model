# News_Prediction_Model

📰 Fake News Prediction Model
📌 Purpose

With the rise of misinformation and fake news online, identifying unreliable news articles has become crucial. The purpose of this project is to build a Machine Learning pipeline that can automatically classify news as real or fake, helping users and organizations filter out misleading information effectively.

📌 Project Overview

This project implements a Machine Learning model using Python to detect fake news. The notebook covers data cleaning, preprocessing, feature extraction, model training, and evaluation, providing a full workflow for reliable news classification.

🛠️ Tech Stack & Tools

Language: Python
Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
Approach: Data cleaning → Preprocessing → Feature extraction → Model training & evaluation

🔹 Data Preprocessing & Feature Engineering

Key steps performed:

Data Cleaning: Removed missing/duplicate entries.
Dataset Merge: Concatenated two datasets for better coverage.
Random Shuffling: Ensured unbiased data distribution.
Text Processing: Tokenization, stopword removal, lowercasing.
Feature Extraction: Converted text into numerical features suitable for ML models.

🔹 Models & Methodology

Used supervised classification algorithms with a standard train-test-split approach:

Logistic Regression
Random Forest Classifier
Decision Tree
Gradient Boosting

Evaluation metrics include accuracy, precision, recall, and F1-score.

📝 Project Approach

This project was approached in a systematic, end-to-end Machine Learning workflow, starting from raw datasets and ending with a trained model capable of classifying news articles as real or fake.

==>Problem Understanding & Objective:

Identified the challenge of misinformation online.
Defined the goal: Build a model that predicts whether a news article is real or fake, providing high accuracy and reliability.

==>Data Collection & Merging:

Acquired two datasets containing labeled news articles.
Merged the datasets to increase coverage and diversity.

==>Data Cleaning & Preprocessing

Removed duplicates and missing values.
Randomly shuffled the data to prevent order bias.
Preprocessed text: lowercasing, tokenization, stopword removal, and punctuation cleaning.

==>Feature Extraction

Converted cleaned text into numerical features suitable for machine learning using techniques such as TF-IDF or Bag-of-Words.
Ensured features capture the semantic patterns of fake vs real news.

==>Model Selection & Training

Applied multiple supervised learning algorithms:

Logistic Regression
Decision Tree
Random Forest Classifier
Gradient Boosting
Split the dataset into training and testing sets to evaluate model generalization.

==>Model Evaluation

Measured performance using accuracy, precision, recall, and F1-score.
Visualized results with confusion matrix and ROC curves.
Compared models to select the best-performing one.

==>Insights & Learnings

Observed which features contribute most to model predictions.
Understood the strengths and weaknesses of different classifiers for text classification tasks.

📢 Future Improvements

Implement deep learning models like LSTM or BERT for better accuracy.
Deploy as an interactive web app for real-time predictions.
Add explainability using SHAP or LIME to interpret predictions.


👤 Author

Yash
LinkedIn (Akash Shaw)
GitHub (Yash8456)
📧 akash.shaw.tiga069@gmail.com
