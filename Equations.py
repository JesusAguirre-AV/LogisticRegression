import pandas as pd
import numpy as np

"""The sigmoid function: maps real value x to a value between 0 and 1
such that they jump tho the likelyhood that x will be the result compared
to other values
This takes the argument x and returns the sigmoid"""
def sigmoid(x):
    return 1/(1+np.exp(-x))
"""Linear predictor, the main intention of this function is to provide a
value for the sigmoid to take in
This requires the vectors of features and corresponding weights (these
should be adjusted as the model grows)"""
def linearPredictor(featuresI, weights):
    return sum(featuresI.insert(0,1) * weights)
"""The predicted probability that a sample belongs to a class
This requires the vectors of features and corresponding weights"""
def hypothesisPredictor(featuresI, weights):
    return sigmoid(linearPredictor(featuresI, weights))
"""Binary-Cross Entropy, measures the difference between predicted
probability and actual label
This requires the target label, the weights, and the features"""
def costFunction(featuresI, weights, target):
    l=len(target)
    s = sigmoid(linearPredictor(featuresI, weights))
    return -(1/l)*(target * np.log(s)+(1-target) * np.log(1-s))
"""The gradient of the cost function
This requires same as its predicesor, being wights, target, and features"""
def gradientCost(featuresI, weights, target):
    l = len(target)
    s = sigmoid(linearPredictor(featuresI, weights))
    error = s-target
    return (1/l) * (featuresI.T @ error)
"""The gaussian distribution function used to predict the feature of a boolean
value of a target
For this we need the mean, variance, and feature"""
def gaussianDistribution(x, mean, variance):
    return np.exp(-((x-mean) ** 2)/(2*variance)) * 1/np.sqrt(2*np.pi*variance)
"""The bayesRuleWGaussian distribution function predict the boolean of the target
based on the features x(vector), the mean values, and the variance (theta squared)"""
def bayesRuleWGaussian(x, mean0, mean1, variance):
    weightO = np.log2((1-np.pi)/np.pi) + sum((mean1**2 - mean0**2)/(2*variance))
    weights = (mean0-mean1)/variance
    return 1/(1 + np.exp(weightO + sum(weights*x)))

"""Next functions start at 3.2 Estimating Parameters for Logistic Regression
"""