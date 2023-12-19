# Used Car Analysis

**Matthew Song**\
Fall 2023

## Overview

### Main Task
Given a training dataset of Used Car sales (Jeep Grand Cherokees and Cadillac XT5s), the task is to build model(s) which can predict the "Trim" (model variant) of a Vehicle (categorical), as well as the listing price (continuous). 

### Models

**The models I choose to build for the Trim Prediction task are:**
* Random Forest (Categorical)
* XGBoost (Categorical)
* Multinomial Logistic Regression

**The models I choose to build for the Price prediction task are:**
* Random Forest (Regression)
* XGBoost (Regression)
* Ridge Regression

I chose to use Random Forest and XGBoost because they are flexible (can be used on categorical and continuous variables) and generally have strong test performance. I also wanted to add in a regression-based alternative, which means I considered Multinomial Logistic Regression for the Trim Prediction task since it is categorical in nature, while I use Ridge Regression for the Price Prediction task since it is continuous in nature. 

### Data

The data provided are: *Training_DataSet.csv* and *Test_Dataset.csv*; however, there is no "answer key" for *Test_Dataset.csv* to compare model predictions off of; it merely serves as a file to read in during the Machine Learning pipeline for producing output predictions.

There are 26 Variables provided in the training dataset--not all of which are useful, and not all of which are pre-cleaned. So, a large part of this project is analyzing the data and implementing cleaning methods to address any data issues.

### Code

I create a program (main.py) which comprises a miniature Machine Learning pipeline from start to finish: 
* Read in the training dataset (*Training_DataSet.csv*)
* Clean and process the training dataset (and the test dataset, which has a similar-but-not-the-same format to the training dataset) 
* Train each model and pick the best-performing model. 
    * Train/Test data split
    * n-Fold Cross Validation with different loss functions. 
    * Hyperparameter tuning via Grid Search. 
* Report model fit statistics
    * Trim Model: ROC AUC (Receiver Operating Characteristic Area Under the Curve)
    * Regression Model: $R^2$
* Apply best model fits onto test data (*Test_Dataset.csv*) to get output predictions (*Test_Output.csv*). 

### Results

**In my model training, XGBoost performed the best for the Trim prediction task, while Random Forest performed the best for the Price prediction task.** 
* For the Trim prediction task, XGBoost attained 66%-72% accuracy in tests, compared to a baseline (predicting the majority class) of 42%-48%, with an ROC AUC score of ~0.85. 
    * This is fairly strong performance, as a +30% (roughly) accuracy over baseline is nontrivial; an ROC AUC score of 0.85 is also quite strong within the literature. 
* For the Price prediction task, Random Forest attained an $R^2$ of 0.78. 
    * Without reading too much into $R^2$, a value of 0.78 is reassuring, as it shows that the model is able to explain 78% of the variance described within the data--typically in projects of similar scope with real-world data, I have often seen much lower $R^2$. 

### Other Information

I utilized Python in this project, relying on the following packages for analysis:
* pandas
* numpy
* sklearn
* xgboost 
* os (helpful to change directories and read in paths)
* time (helpful to time how long models take to run)

### Future Updates

Here are a list of possible future updates that I may try to incorporate into this project:
* Come up with a better name for the project than "Used Car Analysis".
* Deeper dive into Exploratory Data Analysis (EDA)--produce tables and charts to display in the README.
* Text analysis on some variables in the training dataset. 
* Incorporate other ML models for comparison (e.g., SVM, Naive Bayes, etc.)
* Refine Grid Search parameters for potentially better performance. 
* Clean up the code:
    * Move functions outside of Main/into modules, eliminate a few redundant code areas.
    * Split up separate "tasks" in the ML pipeline into separate scripts.
* Create an automated Bash script to run the entire pipeline. 
* Update and provide more discussion/details in the README.

## Data and EDA

To be expanded upon at a future date. 

## Methodology and Code

To be expanded upon at a future date. 

## Results and Next Steps

To be expanded upon at a future date. 