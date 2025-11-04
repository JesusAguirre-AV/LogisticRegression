from __future__ import annotations


def train_svm_rbf(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    print("Starting to train svm")
    clf = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))])
    clf.fit(X, y)
    print("Done training random forest")
    return clf


def train_random_forest(X, y):
    from sklearn.ensemble import RandomForestClassifier
    print("Starting to train random forest")
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X, y)
    print("Done training random forest")
    return clf


def train_gaussian_nb(X, y):
    from sklearn.naive_bayes import GaussianNB
    print("Starting to train naive bayes")
    clf = GaussianNB()
    clf.fit(X, y)
    print("Done training naive bayes")
    return clf


def train_gradient_boost(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    print("Starting to train Gradient Boosting")
    clf = GradientBoostingClassifier()
    clf.fit(X, y)
    print("Done training Gradient Boosting")
    return clf

#TODO
#NEED TO ADD GRADIENT DESCENT AND LOGISTIC REGRESSION BY HAND HERE
#WE WILL NEED TO IMPORT IT INTO MAIN SO WE ARE ABLE TO DO COMPARISON