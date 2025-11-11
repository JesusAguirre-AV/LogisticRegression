# Project 1: Music classification with classical ML

## How to Run

```angular2html
<!--First, load each of the libraries used in the project.-->

<!--main.py will build the database for the features and train the model-->
python main.py
```


## File Manifest
### main.py
Coordinates the full workflow of the project.

-Calls the database-building functions to preprocess and extract features.

-Splits the processed data into training and testing sets.

-Trains multiple models (SVM, Random Forest, Gaussian Naive Bayes, Gradient Boosting, and the group’s custom Logistic Regression).

-Compares model accuracies and generates the Kaggle-style submission.csv file.
### database.py
Handles all aspects of dataset creation and management.

Loads, resamples, and normalizes raw audio files to a consistent sampling rate.

Extracts features according to a configurable FeatureConfig class.

Aggregates features over time and combines them into a unified dataset.

Outputs train_features.csv, test_features.csv, and database.xlsx for transparency and reusability.
### Utils.py
Implements all model-training and mathematical helper functions.

Contains training wrappers for standard models SVM, Random Forest, Gaussian Naive Bayes, and Gradient Boosting.


### LogisticRegressionMultiClass.py
The main implementation of the Logistic Regression model

Contains methods train and predict that handle the grunt work of the actual logistic regression model.



## Contributors
- Benjamin Webster | Wrote Main.py, Database.py, Set up Utils.py
- Gregory Ziter-Glass | Wrote LogisticRegressionMulticlass.py and helped with parameter hypertuning
- Jesus Aguirre | Functions, ...
- Carly Salazar | 

## Performance
- Final Kaggle Score: 71.000
- Accuracy: 73.389
- Date: November 10, 2025

