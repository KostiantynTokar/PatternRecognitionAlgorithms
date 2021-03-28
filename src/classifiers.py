import scipy.optimize as opt
import numpy as np
import math
import util

def trainFishersLinearDiscriminant(class0, class1):
    """

    :param class0: массив векторов первого класса
    :param class1: массив векторов второго класса
    :return: classifier, bias; classifier - функция, принимающая вектор и возвращающая действительное значение:
                               bias - порог классификации такой, что
                               classifier(x) < bias => x in class0
                               classifier(x) >=bias => x in class1
    """
    n0 = class0.shape[0]
    n1 = class1.shape[0]
    m = class0.shape[1]
    X0 = class0
    X1 = class1
    X0mean = np.mean(X0, axis=0)
    X1mean = np.mean(X1, axis=0)

    toSum0 = np.zeros([n0,m,m])
    for i in range(n0):
        toSum0[i] = np.outer(X0[i] - X0mean, X0[i] - X0mean)
    S0 = np.sum(toSum0, axis=0) / (n0 - 1)

    toSum1 = np.zeros([n1,m,m])
    for i in range(n1):
        toSum1[i] = np.outer(X1[i] - X1mean, X1[i] - X1mean)
    S1 = np.sum(toSum1, axis=0) / (n1 - 1)

    Sw = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    w = -np.linalg.inv(Sw).dot(X0mean - X1mean)
    classifier = lambda x: w.dot(x.T)

    bias = 0.5 * classifier(X0mean + X1mean)
    return classifier, bias



def trainNaiveBayes(Nclasses, bagsOfWords, labels, vocabulary):
    """
    Обучает наивный байесовский классификатор
    :param Nclasses: количество классов
    :param bagsOfWords: матрица размерности (количество документов, количество слов в словаре), в которой
                        на позиции [i,j] размещено количество появлений слова j в документе i
    :param labels: массив целых чисел от 0 до Nclasses-1  размерностью (количество документов,)
    :param vocabulary: массив слов, которые появляются в документах
    :return: функция, принимающая bags документов, подлежащих классификации,
             и массив vocab появляющихся в документах слов; функция возвращает массив распределений вероятности,
             в котором на позиции [i,j] размещена вероятность попадения документа i в класс j
    """
    Nbags = bagsOfWords.shape[0]
    Nwords = bagsOfWords.shape[1]
    prior = np.zeros((Nclasses,))
    condprob = np.zeros((Nwords, Nclasses))
    NwordsInClass = np.zeros((Nclasses,))
    for c in range(Nclasses):
        Nc = np.sum(labels == c)
        prior[c] = Nc / Nbags
        bags_c = bagsOfWords[labels == c]
        bag_c = np.sum(bags_c, axis=0)
        NwordsInClass[c] = np.sum(bag_c + 1)
        for t in range(Nwords):
            condprob[t, c] = (bag_c[t] + 1) / NwordsInClass[c]
    def classifier(bags, vocab):
        Ndocs = bags.shape[0]
        NwordsLocal = vocab.shape[0]
        toSum = np.zeros((NwordsLocal,Ndocs,Nclasses))
        for k in range(NwordsLocal):
            index = np.where(vocabulary == vocab[k])
            found = index[0].shape[0] > 0
            if found:
                toSum[k] = np.broadcast_to(bags[:,k].reshape(Ndocs,1), (Ndocs,Nclasses)) * \
                           np.log(condprob[index[0][0],:]).reshape(1,Nclasses)
            else:
                toSum[k] = np.broadcast_to(bags[:,k].reshape(Ndocs,1), (Ndocs,Nclasses)) * \
                           np.log(1 / NwordsInClass).reshape(1,Nclasses)
        res = np.log(prior).reshape(1,Nclasses) + np.sum(toSum, axis = 0)
        res = np.exp(res)
        res = res / np.sum(res, axis = 1).reshape(Ndocs,1)
        return res
    return classifier

def trainBernulli(Nclasses, binaryBagsOfWords, labels, vocabulary):
    """
    Обучает многомерный классификатор Бернулли (multivariate Bernulli model)
    :param Nclasses: количество классов
    :param binaryBagsOfWords: матрица размерности (количество документов, количество слов в словаре), в которой
                              на позиции [i,j] размещена 1, если слово j появляется в документе i, и 0 иначе
    :param labels: массив целых чисел от 0 до Nclasses-1  размерностью (количество документов,)
    :param vocabulary: массив слов, которые появляются в документах
    :return: функция, принимающая binaryBags документов, подлежащих классификации,
             и массив vocab появляющихся в документах слов; функция возвращает массив распределений вероятности,
             в котором на позиции [i,j] размещена вероятность попадения документа i в класс j
    """
    Nbags = binaryBagsOfWords.shape[0]
    Nwords = binaryBagsOfWords.shape[1]
    prior = np.zeros((Nclasses,))
    condprob = np.zeros((Nwords, Nclasses))
    NdocsOfClass = np.zeros((Nclasses,))
    for c in range(Nclasses):
        NdocsOfClass[c] = np.sum(labels == c)
        prior[c] = NdocsOfClass[c] / Nbags
        bags_c = binaryBagsOfWords[labels == c]
        bag_c = np.sum(bags_c, axis=0)
        for t in range(Nwords):
            condprob[t, c] = (bag_c[t] + Nwords) / (NdocsOfClass[c] + 2*Nwords)
    def classifier(binaryBags, vocab):
        Ndocs = binaryBags.shape[0]
        NwordsLocal = vocab.shape[0]
        toSum = np.zeros((NwordsLocal,Ndocs,Nclasses))
        for k in range(NwordsLocal):
            index = np.where(vocabulary == vocab[k])
            found = index[0].shape[0] > 0
            if found:
                toSum[k] = np.broadcast_to(binaryBags[:,k].reshape(Ndocs,1), (Ndocs,Nclasses)) * \
                           np.log(condprob[index[0][0],:]).reshape(1,Nclasses)
            else:
                toSum[k] = np.broadcast_to((1 - binaryBags[:,k]).reshape(Ndocs,1), (Ndocs,Nclasses)) * \
                           np.log(Nwords / (NdocsOfClass + 2*Nwords)).reshape(1,Nclasses)
        res = np.log(prior).reshape(1,Nclasses) + np.sum(toSum, axis = 0)
        res = np.exp(res)
        res = res / np.sum(res, axis = 1).reshape(Ndocs,1)
        return res
    return classifier

def trainSVM(x, y):
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    n = x.shape[0]
    m = x.shape[1]
    left_bounds = np.zeros(n)
    right_bounds = np.full((n,), np.inf)
    bounds = opt.Bounds(left_bounds, right_bounds)
    constr_bound = np.zeros((1,))
    constr_matr = np.expand_dims(y, axis = 0)
    constraint = opt.LinearConstraint(constr_matr, constr_bound, constr_bound)

    x0 = np.zeros_like(y)
    l = opt.minimize(util.lagr, x0, args=(x,y), method='trust-constr',
                     jac='2-point', hess=opt.SR1(),
                     #jac=util.lagr_der, hess=util.lagr_hess,
                     constraints=constraint, bounds=bounds)
    l = l.x
    support_points = np.array([not math.isclose(0.,li,rel_tol=0.,abs_tol=1e-03) for li in l])
    x = x[support_points]
    y = y[support_points]
    l = l[support_points]
    n_supp = x.shape[0]

    lyx = np.expand_dims(l * y, 1) * x
    w = np.sum(lyx, axis=0)
    b = y[0] - np.dot(w,x[0])

    def classifier(input):
        input = np.asarray(input, dtype='float64')
        to_sum = np.dot(lyx, input.T)
        s = np.sum(to_sum, axis=0)
        return np.sign(s + b)
    return classifier
