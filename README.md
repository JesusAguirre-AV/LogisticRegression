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
Builds data base and trains data
### database.py
Makes the database
### Utils.py

### LogisticRegressionMultiClass.py


## Contributors
- Benjamin Webster | Database, ....
- Gregory Ziter-Glass | Main, ...
- Jesus Aguirre | Functions, ...
- Carly Salazar | 

## Performance
- Final Kaggle Score:
- Accuracy: 
- Date: November 10, 2025

