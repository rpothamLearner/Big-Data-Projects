# Do not use anything outside of the standard distribution of python
# when implementing this class
import math 

class LogisticRegressionSGD:
    """
    Logistic regression with stochastic gradient descent
    """

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature
        self.mu = mu

    
#    def fit(self, X, y):
#        """
#        Update model using a pair of training sample
#        """
#        feature_indices = list(dict(X).keys())
#        self.weight = [self.update_weight(i, X, y) if i in feature_indices else self.weight[i] for i in         
#                       range(len(self.weight))]

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        feature_indices = list(dict(X).keys())
        self.weight = [self.update_weight(i, X, y) if i in feature_indices else self.update_weight(i, X, y, 1) for i in                     range(len(self.weight))]

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))


    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if self.predict_prob(X) > 0.5 else 0



    def update_weight(self, i, X, y, flag = 0):
        w_t_1 = self.weight[i]
        if flag == 0:
            x_t = dict(X)[i]
            update = self.eta*((y - self.predict_prob(X))*x_t - 2*self.mu*self.weight[i])
        elif flag == 1:
            update =  -2*self.eta*self.mu*self.weight[i]
        w_t = w_t_1 + update
        return w_t