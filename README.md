README File for Spam and Ham Email Filtration Project

Project Overview

This project aims to develop a machine learning model that can classify emails as either "spam" or "ham" (legitimate) based on their content. The model uses a Multinomial Naive Bayes algorithm to predict the category of an email.

Modules and Libraries Used

Pandas: A popular library for data manipulation and analysis in Python.
Scikit-learn: A machine learning library in Python that provides various algorithms and tools for data analysis and modeling.
CountVectorizer: A module in scikit-learn that converts text data into a matrix of word counts.
MultinomialNB: A module in scikit-learn that implements the Multinomial Naive Bayes algorithm for classification tasks.
train_test_split: A function in scikit-learn that splits a dataset into training and testing sets.
accuracy_score: A function in scikit-learn that calculates the accuracy of a model.
classification_report: A function in scikit-learn that generates a classification report for a model.
confusion_matrix: A function in scikit-learn that generates a confusion matrix for a model.
Dataset

The dataset used in this project is a CSV file containing emails labeled as either "spam" or "ham". The dataset is loaded using the pd.read_csv function from Pandas.

Data Preprocessing

The dataset is preprocessed by:

Grouping the data by category (ham/spam) using the groupby function from Pandas.
Marking ham as 0 and spam as 1 using the apply function from Pandas.
Model Training

The model is trained using the MultinomialNB algorithm from scikit-learn. The training data is split into training and testing sets using the train_test_split function from scikit-learn.

Model Evaluation

The model is evaluated using the following metrics:

Accuracy: calculated using the accuracy_score function from scikit-learn.
Classification Report: generated using the classification_report function from scikit-learn.
Confusion Matrix: generated using the confusion_matrix function from scikit-learn.
Prediction

The model is used to predict the category of new emails. The predicted labels are then printed to the console.

Code Structure

The code is structured into the following sections:

Importing libraries and loading the dataset.
Data preprocessing.
Model training.
Model evaluation.
Prediction.
Running the Code

To run the code, simply execute the Python script in a Python environment with the required libraries installed. The code will load the dataset, preprocess the data, train the model, evaluate the model, and make predictions on new emails.
