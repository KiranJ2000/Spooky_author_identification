# Spooky Author Identification : Project Overview

* An NLP model to identify various authors based on their previous works. The model focuses on 3 authors, namely **Edgar Allan Poe**, **H.P Lovecraft**, and **Mary Wollstonecraft**

* The  dataset, which was provided by kaggle, contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley. The data was prepared by chunking larger texts into sentences using CoreNLP's MaxEnt sentence tokenizer.

* Did Model Ensembling by stacking up **LSTM+GRU**, **Bi-directional LSTM with 1D convolutions**, **Stacked Bi-directional GRU**, **Stacked Convolutions**, and **Stacked Bi-directional LSTM'S** to reach the best model.


## Code and Resources required

**Python Version:** 3.7

**Tensorflow Version:** 2.2

**Packages:** numpy,pandas,sklearn,matplotlib,tensorflow,seaborn


## Data Cleaning

The dataset contained many sentences which had punctuations and words which were not in their root form. The punctuations was removed by using regular expressions and words were converted into their root form by using various libraries such as **WordNetLemmatizer** and **Porter Stemmer**. After cleaning the dataset , was able to find word embeddings for about **88%** of the words in the vocabulary.


## Exploratory Data Analysis( EDA )

Was able to do EDA using matplotlib and seaborn. I used many state of the art EDA techniques such as **WordClouds**, **Topic Modelling**, to name a few. For more insights , do check the 'Exploratory_Data_Analysis' notebook.
 Below are few highlights from my Exploratory Data Analysis.
 
 ![hey](https://user-images.githubusercontent.com/42802226/81496699-18356d80-92d7-11ea-964b-bdf8dbf5b02a.png)   
 
 
 ![download](https://user-images.githubusercontent.com/42802226/81496793-e40e7c80-92d7-11ea-80ee-16066ab1a130.png)
 
 ![download (1)](https://user-images.githubusercontent.com/42802226/81496843-45365000-92d8-11ea-85af-0d892fcf340d.png)


## Model Building

I started of by creating a baseline model which is just a random predictions of the classes, this was followed by creating various machine learning models such as **Logistic Regression**, **Multinomial Naive Bayes**, and **Random Forest** to name a few. 
Then I created various deep learning models such as **Bi-directional LSTM'S** , and **CNN's**. I played around with different combinations of deep learning models and found interesting results. I finally used **Model Ensembling** by stacking various models together. By stacking up models , each and every member of the ensemble makes a contribution to the final output and individual weaknesses are diluted by the contributions from the other members. **For more insights , do check the 'Model_building' notebook.
Below is a table which compares the accuracy of different models**.


| Models                                                               | Accuracy     |     
| -----------------                                                    |:-------------:
| Model Ensembling(Stacking) and Logistic Regression(Meta Learner)     | 88%          |
| Model Ensembling(Weighted Average)                                   | 87%          |   
| Stacked Bi-directional LSTM                                          | 85%          |
| LSTM + GRU                                                           | 85%          |
| Stacked Convolutions-1D                                              | 84%          |
| Bi-directional LSTM with 1D-Convolutions                             | 83%          |
| Stacked Bi-directional GRU                                           | 82%          |
| Multinomial Naive Bayes                                              | 81%          |
| Logistic Regression                                                  | 80%          |
| Support Vector Machines                                              | 79%          |
| Random Forest                                                        | 71%          |
| Random Predictions(Baseline model)                                   | 34%          |






