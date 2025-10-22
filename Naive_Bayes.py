import numpy as np

class NaiveBayes :

  #we don't need __init__ here because we don't have to give any parameters

  def fit(self, X, y):
    n_samples, n_feature = X.shape
    self._classes = np.unique(y)
    n_classes = len(self._classes)

    #calculate mean, variance, prior for each class

    self.__mean = np.zeros((n_classes, n_feature), dtype=np.float64)
    self.__var = np.zeros((n_classes, n_feature), dtype=np.float64)
    self.__prior = np.zeros((n_classes), dtype=np.float64)

    for idx, c in enumerate(self._classes):
      X_c = X[y == c]
      self.__mean[idx, :] = X_c.mean(axis=0)
      self.__var[idx, :] = X_c.var(axis=0)
      self.__prior[idx] = X_c.shape[0] / float(n_samples)


  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)

  def _predict(self,x):
    posteriors = []

    for idx, c in enumerate(self._classes):
      prior = np.log(self.__prior[idx])
      posterior = np.sum(np.log(self._pdf(idx,x)))
      posterior = posterior +prior
      posteriors.append(posterior)

    return self.__classes[np.argmax](posteriors)
  
  def _pdf(self, class_idx, x):
    mean = self.__mean[class_idx]
    var = self.__var[class_idx]
    numerator = np.exp(-((x - mean)**2) / (2 * var))
    denominator = np.sqrt(2 * np.pi * var)
    return numerator / denominator