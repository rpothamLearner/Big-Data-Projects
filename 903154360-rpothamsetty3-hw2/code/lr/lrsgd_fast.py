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
        self.hash = [0.0] * n_feature
        self.k = 0


    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        feature_indices = list(dict(X).keys())
        non_zero = { k:v for k, v in dict(X).items() if v != 0 }
        features_nonz =  list(non_zero.keys())
        self.k += 1
        
        self.weight = [self.update_weight(j, X, y) if j in features_nonz else self.weight[j] for j in range(len(self.hash))]
        
        self.hash = [self.k if i in features_nonz else self.hash[i] for i in range(len(self.hash))]
        

    def fit_endweight(self, X, y):
        """
        Update weights for zero-valued features with non-zero count in the hash table
        To be used after the last datum
        """
        self.weight = [self.update_weight(j, X, y, 1) for j in range(len(self.hash))]
        

    def predict_prob(self, X):
        """
        Sigmoid function
        """
        try:
            return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
        except:
            print(len(self.weight))
            #print(X)


    def predict(self, X):
        """
        Predict 0 or 1 given X and the current weights in the model
        """
        return 1 if self.predict_prob(X) > 0.5 else 0



    def update_weight(self, j, X, y, flag = 0):
        if flag == 0:
            x_t = dict(X)[j]
            upd1 = self.weight[j]*(1 - 2*self.eta*self.mu)**(self.k - self.hash[j])
            w_t = upd1 + self.eta*(y - self.predict_prob(X))*x_t
        elif flag == 1:
            b = (self.k - self.hash[j])
            a = ((1 - 2*self.eta*self.mu)**b)
            w_t = self.weight[j]*a
        return w_t