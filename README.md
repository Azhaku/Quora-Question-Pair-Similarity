# Quora Question Pair Similarity
This project is aimed at identifying duplicate questions on Quora using machine learning techniques. The goal is to help Quora identify duplicate questions and reduce the number of redundant questions on the platform.

## Dataset
The dataset for this project contains pairs of questions from Quora and labels indicating whether the questions are duplicates or not. The dataset can be downloaded from [here](https://www.kaggle.com/c/quora-question-pairs/data).

## Approach
The approach used in this project involves the following steps:

1) Data cleaning and preprocessing
2) Feature engineering
3) NLP Techniques
4) Training and testing machine learning models
5) Model selection and tuning
6) Evaluation of the best-performing model
7) Deployment on heroku
8) MLOps
We used various natural language processing techniques(BOW,TFIDF,Glove and Sendence BERT) to preprocess and engineer features from the text data. We then trained several supervised machine learning models, including logistic regression, decision tree, random forest,KNN, and XGBoost, to predict whether a pair of questions are duplicates. We evaluated the models using standard classification metrics such as accuracy, precision, recall, and F1 score.

## Results
Our best-performing model achieved an F1 score of 0.86 on the test set, which indicates a high degree of accuracy in identifying duplicate questions on Quora.
