from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from database import FeatureConfig, build_train_dataframe, build_test_dataframe, \
    save_database_artifacts
from Utils import train_svm_rbf, train_random_forest, train_gaussian_nb, train_gradient_boost
from LogisticRegression import LogisticRegressionMultiClass

#Figured out way to have file path work
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


"""**************************************************************** Parameters ***************************************************************************"""
mfcc = True
fcc_delta=False
chroma=True
spectral_contrast=True
zcr=True
spectral_centroid=True
spectral_bandwidth=True
spectral_rolloff=True
rms=True
tempo=False
n_mfcc=20
aggregation="mean_std"

#How far we move
logistRegressStepSize = 0.1
#How many iterations of training
logistRegressEpochs = 300
"""*******************************************************************************************************************************************************"""


#Change these paramters to try a fuckton of different things,
def build_database():
    print("Configuring features...")
    cfg = FeatureConfig(
        mfcc=True,
        mfcc_delta=False,
        chroma=True,
        spectral_contrast=True,
        zcr=True,
        spectral_centroid=True,
        spectral_bandwidth=True,
        spectral_rolloff=True,
        rms=True, tempo=False,
        n_mfcc=20,
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
    X = df_tr.drop(columns=["label","path"]).to_numpy(dtype=float)
    y_text = df_tr["label"].astype(str).to_numpy()
    le = LabelEncoder().fit(y_text)
    y = le.transform(y_text)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    #train comparison models
    gbm = train_gradient_boost(Xtr, ytr)
    svm = train_svm_rbf(Xtr, ytr)
    rf  = train_random_forest(Xtr, ytr)
    gnb = train_gaussian_nb(Xtr, ytr)

    lr = LogisticRegressionMultiClass(logistRegressStepSize, LogisticRegressEpochs)
    lr.train(Xtr, ytr)

    pred_gbm = gbm.predict(Xte); acc_gbm = accuracy_score(yte, pred_gbm)
    pred_svm = svm.predict(Xte); acc_svm = accuracy_score(yte, pred_svm)
    pred_rf  = rf.predict(Xte);  acc_rf  = accuracy_score(yte, pred_rf)
    pred_gnb = gnb.predict(Xte); acc_gnb = accuracy_score(yte, pred_gnb)
    pred_lr = lr.predict(Xte)

    print("\nAccuracies")
    print(f"GBM                : {acc_gbm:.4f}")
    print(f"SVM (rbf)          : {acc_svm:.4f}")
    print(f"RandomForest       : {acc_rf:.4f}")
    print(f"GaussianNB         : {acc_gnb:.4f}")

    results = {
        "gbm": acc_gbm,
        "svm_rbf": acc_svm,
        "random_forest": acc_rf,
        "gaussian_nb": acc_gnb,
    }

    #Just added this for testing purposes we will return the
    results = {"svm_rbf": acc_svm, "random_forest": acc_rf, "gaussian_nb": acc_gnb}
    best_name = max(results, key=results.get)
    if best_name == "svm_rbf":        best_model = ("svm", svm)
    elif best_name == "random_forest": best_model = ("rf",  rf)
    elif best_name == "gbm": best_model = ("gbm", gbm)
    else:                              best_model = ("gnb", gnb)

    print("Best model on validation", best_name)
    return best_name, best_model, le


def make_kaggle_submission(df_te: pd.DataFrame, model_info, label_encoder, out_csv: Path):
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
    #build database edit function build database for different features
    #(I have no idea what a lot of those features do was just built into a libary)
    print("Starting to build database")
    train_df, test_df = build_database()

    #Train models and compare
    #TODO need to add LR to this function
    print("Starting to train models")
    name, model_info, label_encoder = train_and_compare(train_df)

    #Makes kaggle submission in LogisticRegression\data\processed
    make_kaggle_submission(test_df, model_info, label_encoder,
                           PROC / "submission.csv")
