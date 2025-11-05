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
"""The bayesRuleWGaussian distribution function predict the boolean of the target
based on the features x(vector of the same training level and target), the mean values,
and the variance (theta squared)"""
def bayesRuleWGaussian(features, mean0, mean1, variance):
    weightO = np.log2((1-np.pi)/np.pi) + sum((mean1**2 - mean0**2)/(2*variance))
    weights = (mean0-mean1)/variance
    return 1/(1 + np.exp(weightO + sum(weights*features)))

"""Weight update for the training process that updates the whole array of weights based
on how accurate or inaccurate they may be in terms of providing a reliable regression
takes in the features, targets, weights, desired step size, and the predicted parameters"""
def weight_update(features, targets, weights, stepSize, LRprediction):

    mean0 = meanFunction(0, targets, features)
    mean1 = meanFunction(1, targets, features)
    variance0 = varianceFunction(0, features, targets, mean0)
    variance1 = varianceFunction(1, features, targets, mean1)

    error = targets - LRprediction
    summation = features.T @ error

    return weights + stepSize*summation

"""This function is meant to find the mean values used in the Gaussian NB function, designed
to create a mean dependant on the label we are targeting
The arguments are the features(should be a matrix based on index and training level), target label, and the targets"""
def meanFunction(label, targets, features):
    targetMask = [1 if i == label else 0 for i in targets]
    return (1/sum(targetMask))* (features.T @ targetMask)

"""This is the function used to create a vector of the variance based on their desired label,
targets features, target, and the previous mean values"""
def varianceFunction(label, features, targets, mean):
    targetMask = [1 if i == label else 0 for i in targets]
    errorMatrixSquare = (features - mean) ** 2
    return ((1/sum(targetMask))-1) * (errorMatrixSquare.T @ targetMask)

"""Making a vector such that it returns the argmax for the weights vector needed in the weight update function
for this we need the features related to their classifications in sets such that their probabilities are
dependant on weights"""
#TODO

"""Here is the optimization function, not found in the reading for logistic regresion though it is in the slides,
it seems to be meant to """
#TODO

"""Functions for discrete values, in the case where Y is {y1, ..., yn} rather than a boolean, however a boolean is
what we"""