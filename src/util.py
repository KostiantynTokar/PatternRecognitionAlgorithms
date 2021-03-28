import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import ROC
import math
import classifiers

def classifierROC(classifier, class0, class1):
    scores0 = classifier(class0)
    scores1 = classifier(class1)
    scores = np.concatenate((scores0, scores1), axis=0)
    labels0 = np.zeros([scores0.shape[0]])
    labels1 = np.ones([scores1.shape[0]])
    labels = np.concatenate((labels0, labels1), axis=0)
    tpr, fpr, thresholds = ROC.calcROC(labels, scores)
    return tpr, fpr, thresholds

def extractDocsAndLabels(path):
    dataFrame = pd.read_csv(path)
    return dataFrame.values[:,1], dataFrame.values[:,0]

def preprocessDocs(docs):
    cv = CountVectorizer()
    bagsOfWords = cv.fit_transform(docs).toarray()
    return np.array(bagsOfWords), np.array(cv.get_feature_names())

def getDataset(bagsOfWords, labels, trainPercentage = 0.8):
    ham = bagsOfWords[labels == 'ham']
    spam = bagsOfWords[labels == 'spam']
    nHam = ham.shape[0]
    nSpam = spam.shape[0]
    nHamTrain = math.trunc(nHam * trainPercentage)
    nSpamTrain = math.trunc(nSpam * trainPercentage)

    hamTrain = ham[:nHamTrain]
    spamTrain = spam[:nSpamTrain]
    bagsTrain = np.concatenate((hamTrain, spamTrain), axis=0)
    labelsTrain = np.concatenate((['ham'] * nHamTrain, ['spam'] * nSpamTrain))

    hamTest = ham[nHamTrain:]
    spamTest = spam[nSpamTrain:]
    bagsTest = np.concatenate((hamTest, spamTest), axis=0)
    labelsTest = np.concatenate((['ham'] * (nHam - nHamTrain), ['spam'] * (nSpam - nSpamTrain)))

    return bagsTrain, labelsTrain, bagsTest, labelsTest

def trainBinaryNaiveBayes(bagsOfWords, labels, vocabulary):
    NB = classifiers.trainNaiveBayes(2, bagsOfWords, labels, vocabulary)
    return lambda bags, vocab: NB(bags,vocab)[:,0]

def trainBinaryBernulli(binaryBagsOfWords, labels, vocabulary):
    B = classifiers.trainBernulli(2, binaryBagsOfWords, labels, vocabulary)
    return lambda bags, vocab: B(bags,vocab)[:,0]

def lagr(l, x, y):
    """
    Функия Лагранжа, которую нужно минимизировать по l при фиксированных x, y
    :param l: множители Лагранжа
    :param x: массив входных векторов
    :param y: массив меток -1 и 1
    :return: значение функции Лагранжа
    """
    l = np.asarray(l, dtype='float64')
    y = np.asarray(y, dtype='float64')
    x = np.asarray(x, dtype='float64')
    n = x.shape[0]
    m = x.shape[1]
    prod = np.expand_dims(l * y, 1) * x
    matr1 = np.broadcast_to(np.expand_dims(prod, 0), (n,n,m))
    matr2 = np.broadcast_to(np.expand_dims(prod, 1), (n,n,m))
    matr_prod = matr1 * matr2
    matr = np.sum(matr_prod, axis = 2)
    #matr = np.tensordot(np.broadcast_to(np.expand_dims(prod, 0), (n,n,m)),
    #                    np.broadcast_to(np.expand_dims(prod, 1), (n,n,m)), axes=([2],[2]))
    return - np.sum(l) + 0.5 * np.sum(matr)

def lagr_der(l, x, y):
    l = np.asarray(l, dtype='float64')
    y = np.asarray(y, dtype='float64')
    x = np.asarray(x, dtype='float64')
    n = x.shape[0]
    m = x.shape[1]
    prod = np.expand_dims(l * y, 1) * x
    prod_der = np.expand_dims(y, 1) * x
    matr1 = np.broadcast_to(np.expand_dims(prod, 0), (n,n,m))
    matr2 = np.broadcast_to(np.expand_dims(prod_der, 1), (n,n,m))
    matr_prod = matr1 * matr2
    matr = np.sum(matr_prod, axis=2)
    double_diag = np.diag(np.ones_like(l))
    double_diag += np.ones_like(double_diag)
    matr *= double_diag
    return -1 + 0.5 * np.sum(matr, axis=1)

def lagr_hess(l, x, y):
    l = np.asarray(l, dtype='float64')
    y = np.asarray(y, dtype='float64')
    x = np.asarray(x, dtype='float64')
    prod_der = np.expand_dims(y, 1) * x
    matr = np.dot(prod_der, prod_der.T)
    double_diag = np.diag(np.ones_like(l))
    double_diag += np.ones_like(double_diag)
    matr *= double_diag
    return 0.5 * matr

class Line:
    def __init__(self, points = None, coeffs = None):
        self.a = 0
        self.b = 0
        self.c = 0
        if points is not None:
            x1 = points[0]
            x2 = points[1]
            self.a = x1[1] - x2[1]
            self.b = x2[0] - x1[0]
            self.c = x1[0] * x2[1] - x2[0] * x1[1]
        elif coeffs is not None:
            self.a = coeffs[0]
            self.b = coeffs[1]
            self.c = coeffs[2]
        self._normalize()
    def rel_dist(self, x):
        d = np.sqrt(self.a**2 + self.b**2)
        return self.a * x[0] + self.b * x[1] + self.c / d
    def intersection(self, l):
        d = self.a*l.b-l.a*self.b
        return np.array([-(self.c*l.b-l.c*self.b)/d, -(self.a*l.c-l.a*self.c)/d])
    def build_parallel(self, x):
        return Line(coeffs = [self.a, self.b, -self.a * x[0] - self.b * x[1]])
    def build_perpend(self, x):
        return Line(coeffs = [-self.b, self.a, -self.a * x[1] + self.b * x[0]])
    def angle_coef(self):
        if self.b==0:
            return np.pi
        return -self.a #b==0 due to normalization
    def _normalize(self):
        if self.b!=0:
            self.a/=self.b
            self.c/=self.b
            self.b=1
        elif self.a!=0:
            self.c/=self.a
            self.a=1
        else:
            self.c=0

def rotation_matrix(dim, i, j, alpha):
    res = np.eye(dim)
    res[i,i] = np.cos(alpha)
    res[i,j] = -np.sin(alpha)
    res[j,i] = np.sin(alpha)
    res[j,j] = np.cos(alpha)
    return res
