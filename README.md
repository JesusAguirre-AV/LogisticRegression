# Project 2: Music classification with classical ML

## How to Run
1. Include the audio files dataset, copy the `test` and `train` folders under the `processed` folder.
```angular2html
<!-- Project Folder Structure -->
.
├── LogisticRegressionMultiClass.py
├── README.md
├── Utils.py
├── __pycache__
│   ├── LogisticRegressionMultiClass.cpython-313.pyc
│   ├── Utils.cpython-313.pyc
│   └── database.cpython-313.pyc
├── data
│   ├── processed
│   │   ├── database.xlsx
│   │   ├── label_map.json
│   │   ├── submission.csv
│   │   ├── test_features.csv
│   │   └── train_features.csv
│   └── raw <!-- Place music folders here under raw -->
│       ├── test
│       └── train
├── database.py
├── main.py
└── sample_sub.csv
```
2. Load each of the libraries used in the project.
3. Build the database and train the models.
```angular2html
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

