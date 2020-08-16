import numpy as np
"""
Naive Bayes:
   posterior      prior    likelihood
 P(y(k) | x) = [ P(y(k)) * P(x | y(k)) ] / P(x)

Smoothing processing:
   P(y(k)) = ( n(y(k)) + a ) / ( n + m * a)
   Def. n(y(k): # of specific label
        n: total # of all labels
        a: smoothing parameter, alpha
        m: # types of labels
"""

class Naive_Bayes():
    def __init__(self, alpha):
        self.label_probs = dict()
        self.label_statistics = dict()
        self.alpha = alpha
    
    def prepare_data(self):
        features = np.array(
            [
                [320, 204, 198, 265],
                [253, 53, 15, 2243],
                [53, 32, 5, 325],
                [63, 50, 42, 98],
                [1302, 523, 202, 5430],
                [32, 22, 5, 143],
                [105, 85, 70, 322],
                [872, 730, 840, 2762],
                [16, 15, 13, 52],
                [92, 70, 21, 693]
            ])
        labels = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        return features, labels
    
    # Gaussian Model
    def cal_mu_and_sigma(self, feature):
        mu = np.mean(feature)
        sigma = np.std(feature)
        return (mu, sigma)
    
    def train(self, features, labels):
        ''' 
        label_probs <dict>
          key=1: abnormal user 
          key=0: normal user
        '''
        label_types = set(labels)
        # Smoothing processing
        self.label_probs[1] = (sum(labels) + self.alpha) / (len(labels) + len(label_types) * self.alpha)
        self.label_probs[0] = 1 - self.label_probs[1]
        print("label_probs:\n", self.label_probs, '\n')
        
        '''
        label_statistics <dict>
          [example]
          { label_A: { feature_A: {mu:xx, sigma:xx}, 
                       feature_B: {mu:xx, sigma:xx},
                       ...
                     },
            label_B: { ... }, 
            ...
          }
        '''
        feature_dim = len(features[0]) # dim: dimension
        for label_type in label_types: # in this case, label_type∈[0,1]
            self.label_statistics.setdefault(label_type, dict())    
            label_type_features = features[np.equal(labels, label_type)]
            print("label:", label_type)
            print("feature:")
            for axis in range(feature_dim):
                print(f"  (axis={axis})")
                feature = label_type_features[:, axis]
                print("    ", feature, sep='')
                self.label_statistics[label_type].setdefault(axis, self.cal_mu_and_sigma(feature))
            print()
        print("label_statistics:")
        print(self.label_statistics, '\n')
    
    # Calculating likelihood: P(x(i)|y(k))
    def cal_gaussian_funcVal(self, mu, sigma, x):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x-mu)**2 / (2 * sigma**2))
    
    # Predict whether the given user is a abnormal user
    def predict(self, test_features):
        # transformat into numpy array
        test_features = np.array(test_features)
        
        # initial setting
        pred_label = None
        maxP = 0
        
        # Calculate and update:
        #    until obtains the maximum of P(y(k)|x)
        for label_type, label_prob in self.label_probs.items():
            #print("probability of label {}: {}".format(label_type, label_prob))
            
            feature_probs = self.label_statistics[label_type]
            #print(feature_probs)
            
            continuous_product = 1.0
            for axis, mu_and_sigma in feature_probs.items():
                mu, sigma = mu_and_sigma[0], mu_and_sigma[1]
                prob_likelihood = self.cal_gaussian_funcVal(mu, sigma, test_features[axis])
                continuous_product *= prob_likelihood 
            
            curr_result = label_prob * continuous_product
            if curr_result > maxP:
                maxP = curr_result
                pred_label = label_type
        #print()
        return pred_label
    
    def trans_predLabel_to_result(self, pred_label):
        if pred_label == 0:
            return "normal user"
        elif pred_label == 1:
            return "abnormal user"
                
if __name__ == "__main__":
    nb = Naive_Bayes(alpha=1.0)
    features, labels = nb.prepare_data()
    nb.train(features, labels)
    
    feature_names = ["Registration Days", "Active Days", "Number of Purchases", "Number of Clicks"]
    test_features = [134, 84, 235, 349]
    pred_label = nb.predict(test_features)
    result = nb.trans_predLabel_to_result(pred_label)
    print("If the user's features with {} is {},".format(tuple(feature_names), tuple(test_features)))
    print("according to the prediction of naive bayes classifier,")
    print("this user is most probably \"{}\".".format(result))
    