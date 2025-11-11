from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time

from database import FeatureConfig, build_train_dataframe, build_test_dataframe, \
    save_database_artifacts
from Utils import train_svm_rbf, train_random_forest, train_gaussian_nb, train_gradient_boost
from LogisticRegressionMultiClass import LogisticRegressionMultiClass

# Figured out way to have file path work
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

"""**************************************************************** Parameters ***************************************************************************"""
mfcc = False
fcc_delta = True  # <--
chroma = False
spectral_contrast = False  #
zcr = False
spectral_centroid = False  #
spectral_bandwidth = False
spectral_rolloff = True  # <--
rms = True  # <--
tempo = False  #
n_mfcc = 5  # <--
aggregation = "mean_std"

# How far we move
logistRegressStepSize = 0.3
# How many iterations of training
logistRegressEpochs = 200
"""*******************************************************************************************************************************************************"""

def build_database():
    """
    Builds a database for the features, so the models can train on structured data.
    :return: A tuple (df_tr, df_te) containing the training and test data frames.
    """
    print("Configuring features...")
    cfg = FeatureConfig(
        mfcc=False,
        mfcc_delta=True,
        chroma=False,
        spectral_contrast=False,
        zcr=False,
        spectral_centroid=False,
        spectral_bandwidth=False,
        spectral_rolloff=True,
        rms=True,
        tempo=False,
        n_mfcc=5,
        aggregation="mean_std",
    )
    print("Features configured, building training dataframe")
    df_tr = build_train_dataframe(str(RAW / "train"), cfg)
    print("Training dataframe built, building test dataframe")
    df_te = build_test_dataframe(str(RAW / "test"), cfg)
    print("Test dataframe built, saving database artifacts")
    save_database_artifacts(df_tr, df_te, str(PROC))
    print("Done")
    return df_tr, df_te


def train_and_compare(df_tr: pd.DataFrame):
    """
    This trains each of the different models, including: gradient boosting, support vector machines, random forest,
     Gaussian Naive Bayes, and logistic regression.
    :param df_tr: Training data frame
    :return: the name of the best model, best model info, label encoder
    """
    X = df_tr.drop(columns=["label", "path"]).to_numpy(dtype=float)
    y_text = df_tr["label"].astype(str).to_numpy()
    le = LabelEncoder().fit(y_text)
    y = le.transform(y_text)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # training time for each of the models
    training_times = {}

    # train comparison models
    start_time = time.perf_counter()
    gbm = train_gradient_boost(Xtr, ytr)
    training_times['gbm'] = time.perf_counter() - start_time

    start_time = time.perf_counter()
    svm = train_svm_rbf(Xtr, ytr)
    training_times['svm_rbf'] = time.perf_counter() - start_time

    start_time = time.perf_counter()
    rf = train_random_forest(Xtr, ytr)
    training_times['random_forest'] = time.perf_counter() - start_time

    start_time = time.perf_counter()
    gnb = train_gaussian_nb(Xtr, ytr)
    training_times['gnb'] = time.perf_counter() - start_time

    # these two lines below are suspicious
    start_time = time.perf_counter()
    lr = LogisticRegressionMultiClass(logistRegressStepSize, logistRegressEpochs)
    lr.train(Xtr, ytr)
    training_times['lr'] = time.perf_counter() - start_time

    # gradient boosting machine
    pred_gbm = gbm.predict(Xte)
    acc_gbm = accuracy_score(yte, pred_gbm)
    # support vector machines
    pred_svm = svm.predict(Xte)
    acc_svm = accuracy_score(yte, pred_svm)
    # random forest
    pred_rf = rf.predict(Xte)
    acc_rf = accuracy_score(yte, pred_rf)
    # Guassian naive bayes
    pred_gnb = gnb.predict(Xte)
    acc_gnb = accuracy_score(yte, pred_gnb)
    # logistic regression
    pred_lr = lr.predict(Xte)
    acc_lr = accuracy_score(yte, pred_lr)

    print("\nAccuracies")
    print(f"GBM                : {acc_gbm:.4f}")
    print(f"SVM (rbf)          : {acc_svm:.4f}")
    print(f"RandomForest       : {acc_rf:.4f}")
    print(f"GaussianNB         : {acc_gnb:.4f}")
    print(f"LogisticRegression : {acc_lr:.4f}")

    print("\nTraining Times (seconds)")
    print(f"GBM                : {training_times['gbm']:.4f}")
    print(f"SVM (rbf)          : {training_times['svm_rbf']:.4f}")
    print(f"RandomForest       : {training_times['random_forest']:.4f}")
    print(f"GaussianNB         : {training_times['gnb']:.4f}")
    print(f"LogisticRegression : {training_times['lr']:.4f}")

    # commented out to just get logistic regression
    results = {
        # "gbm": acc_gbm,
        # "svm_rbf": acc_svm,
        # "random_forest": acc_rf,
        # "gnb": acc_gnb,
        "lr": acc_lr,
    }

    best_name = max(results, key=results.get)
    if best_name == "svm_rbf":
        best_model = ("svm_rbf", svm)
    elif best_name == "random_forest":
        best_model = ("random_forest", rf)
    elif best_name == "gbm":
        best_model = ("gbm", gbm)
    elif best_name == "lr":
        best_model = ("lr", lr)
    elif best_name == "gnb":
        best_model = ("gnb", gnb)
    else:
        # This code should never be reached, but it prevents the error
        print(f"Error: {best_name}. Default to gnb.")
        best_model = ("gnb", gnb)

    print("Best model on validation", best_name)
    return best_name, best_model, le


def make_kaggle_submission(df_te: pd.DataFrame, model_info, label_encoder, out_csv: Path):
    """
    Creates a CSV for Kaggle submission
    :param df_te: testing data frame
    :param model_info: information about the model for training
    :param label_encoder: converts numeric labels back to class names
    :param out_csv: Path to CSV
    """
    tag, model = model_info
    X_test = df_te.drop(columns=["path"]).to_numpy(dtype=float)
    yhat = model.predict(X_test)
    preds = label_encoder.inverse_transform(yhat)

    # id = basename of file path
    ids = df_te["path"].apply(lambda p: Path(p).name).tolist()
    sub = pd.DataFrame({"id": ids, "class": preds})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"[ok] wrote Kaggle submission {out_csv} (rows={len(sub)})")


if __name__ == "__main__":
    """
    Builds database and run model testing/training
    """
    # build database edit function build database for different features
    print("Starting to build database")
    train_df, test_df = build_database()

    # Train models and compare
    print("Starting to train models")

    # Find best model
    best_name, _, label_encoder = train_and_compare(train_df)

    print(f"\nValidation complete. Best model type is: {best_name}")
    print("Retraining best model on 100% of the training data...")

    # Use full training dataset for final model
    X_full = train_df.drop(columns=["label", "path"]).to_numpy(dtype=float)
    y_full_text = train_df["label"].astype(str).to_numpy()
    # must use the *same* label encoder, so we'll re-fit it
    le_full = LabelEncoder().fit(y_full_text)
    y_full = le_full.transform(y_full_text)

    # Initialize a final_model variable
    final_model = None

    # Use the best_name to create and train the final model
    if best_name == "svm_rbf":
        final_model = train_svm_rbf(X_full, y_full)
    elif best_name == "random_forest":
        final_model = train_random_forest(X_full, y_full)
    elif best_name == "gbm":
        final_model = train_gradient_boost(X_full, y_full)
    elif best_name == "lr":
        final_model = LogisticRegressionMultiClass(logistRegressStepSize, logistRegressEpochs)
        final_model.train(X_full, y_full)
    elif best_name == "gnb":
        final_model = train_gaussian_nb(X_full, y_full)

    print("Final model retrained.")

    # Package the final model info
    final_model_info = (best_name, final_model)

    print(final_model_info)

    # Makes kaggle submission in LogisticRegression\data\processed
    make_kaggle_submission(test_df, final_model_info, le_full,
                           PROC / "submission.csv")