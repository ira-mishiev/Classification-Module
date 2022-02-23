# Classification-Module
Classification Project Write-up

Identifying patients with a risk of cardio disease

Abstract

Heart Disease is one of the most frequently appearing chronic diseases in the United States. The ability to diagnose the disease early and understand the factors that lead to cardio diseases would help prevent the development of the illness for thousands of people. This project aims to predict the individuals that might have cardio disease by using medical records after regular visits to a physician. 

Design

The project is focusing on building a classification model using data from [Kaggle](https://www.kaggle.com/raminhashimzade/cardio2model/notebook) that consists of medical records of 70000 people with identification whether this individual has developed a heart disease or not. Predicting the development of illness accurately using classification models would enable physicians to take action before the disease is in place. 

Data

The dataset contains medical records of 70000 people, with 11 columns describing the medical description for each individual. Every raw includes records for gender, age, height, weight, systolic and diastolic blood pressure, glucose level, alcohol consumption, physical activity, smoking experience, and whether this individual has heart disease or not. 


Algorithms

Feature Engineering

Baseline - Logistic Regression

Model Evaluation and Selection
The entire dataset was split into 80/20/20 train, validation, and holdout data. Scores reported below were only calculated on the validation part, and predictions were made on the 20% holdout. The recall was used to measure the model's accuracy because for this project is more important to identify correctly positive class. 

Random Forest accuracy after tuning hyperparameters:

Test set: 72.96%
Recall score 77.03%
ROC AUC score 0.72

Tools

Data Manipulation: Numpy and Pandas 
Modeling: Scikit learn 
Plotting: Matplotlib & Seaborn

Communication

The output of this project is showcased in the presentation





