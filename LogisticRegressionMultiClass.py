import numpy as np
import time
#from Utils import predictProbabilitiesFromWeights

class LogisticRegressionMultiClass:
    def __init__(self, stepSize=0.1, epochs=300, regLambda=0.0001, batchSize=256):
        self.stepSize = stepSize
        self.epochs = epochs
        self.regLambda = regLambda
        self.batchSize = batchSize

        self.weights = None
        self.bias = None
        self.mean = None
        self.sd = None

        self.rng = np.random.default_rng(int(time.time()))

    """Logits calculation"""
    def logits(self, fs):
            return (fs @ self.weights.T + self.bias)


    """Softmax calculation (turning logits into valid probabilities)"""
    def softmax(self, Z):
        Zs = Z - np.max(Z, axis=1, keepdims=True)
        E = np.exp(Zs)
        return E / np.sum(E, axis=1, keepdims=True)

    def logitsAndSoftmax(self, xs):
        return self.softmax(self.logits(xs))

    """Forward pass to get probabilities for weight_update"""
    def predictProbabilitiesFromWeights(self, fs):
        return self.logitsAndSoftmax(fs)


    """argmax vector calculation"""
    def argmaxVector(self, fs):
        return np.argmax(self.predictProbabilitiesFromWeights(fs), axis=1)


    def oneHot(self, y, K):
        Y = np.zeros((y.shape[0], K), dtype=float)
        Y[np.arange(y.shape[0]), y] = 1.0
        return Y


    def _stdFit(self, X):
        self.mean = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        return(X - self.mean) / self.sd


    def _stdApply(self, X):
        return (X - self.mean)/ self.sd


    """Training function"""
    def train(self, X, y):
        X = X.astype(float, copy=False)
        y = y.astype(int, copy =False)

        Xn = self._stdFit(X)
        N, D = Xn.shape
        K = int(y.max()) + 1
        Y = self.oneHot(y, K)

        self.weights = self.rng.normal(0, 0.01, size=(K, D))
        self.bias = np.zeros(K)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(N)
            for start in range(0, N, self.batchSize):
                end = min(start+self.batchSize, N)
                batch = idx[start:end]
                Xb, Yb = Xn[batch], Y[batch]

                P = self.logitsAndSoftmax(Xb)

                B = Xb.shape[0]
                diff = (P- Yb) / B
                gradientWeights = diff.T @ Xb + self.regLambda * self.weights
                gradientBias = diff.sum(axis=0)

                self.weights -= self.stepSize * gradientWeights
                self.bias -= self.stepSize * gradientBias
        return self



    """Predict function"""
    def predict(self, X):
        preds = self._stdApply(X.astype(float, copy=False))
        return self.argmaxVector(preds)