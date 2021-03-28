import numpy as np
import matplotlib.pyplot as plt

def calcROC(labels, scores):
    """
    :param labels: np.array (shape = [n_samples]), состоящий из 0 и 1, где 0 отвечает Negative, а 1 - Positive
    :param scores: np.array (shape = [n_samples]), состоящий из значений порога, при переходе через который
                                                   объект будет классицифирован как Positive
    :return: tpr, fpr, thresholds - True Positive Rate (sensitivite, чувствительность) и False Positive Rate
                                    (specificity, специфичность) в порядке возростания,
                                    thresholds в порядке убывания (уменьшается thresholds -> увеличиваются
                                    TP и FP, уменьшаются FN и TN -> увеличиваются tpr и fpr)
    """
    pairs = np.stack((labels, scores), axis=0)
    pairs = pairs[:, pairs[1].argsort()]
    uniqueScores = np.unique(pairs[1]) #уникальные пороги
    size = uniqueScores.shape[0]
    pairs, counts = np.unique(pairs, return_counts=True, axis=1)
    tableOfCounts = np.zeros([3,size]) #tableOfCounts[2] - уникальные пороги, tableOfCounts[0] и tableOfCounts[1] -
                                       #количество объектов класса 0 и 1 соответственно, которые имеют соотв. порог
    tableOfCounts[2] = uniqueScores
    for (pair,count) in zip(pairs.T,counts):
        tableOfCounts[int(pair[0]), uniqueScores == pair[1]] += count
    thresholds = np.flip(uniqueScores, axis=0)
    thresholds = np.concatenate(([uniqueScores[-1]+1.], thresholds), axis=0)
    tpr = np.empty([size + 1])
    fpr = np.empty([size + 1])

    #От максимального порога, когда все объекты классифицируются как Negative (0),
    # до минимального, когда все объекты Positive (1).
    tpr[0] = 0.
    fpr[0] = 0.
    TN = (labels == 0).sum()
    FN = (labels == 1).sum()
    TP = 0
    FP = 0

    for i in range(0, size):
        count0 = tableOfCounts[0, size - i - 1]
        count1 = tableOfCounts[1, size - i - 1]
        #Было FN, стало TP
        TP += count1
        FN -= count1
        #Было TN, стало FP
        FP += count0
        TN -= count0
        tpr[i + 1] = TP / (TP + FN)
        fpr[i + 1] = FP / (FP + TN)

    return tpr, fpr, thresholds

def AUC(tpr, fpr):
    res = 0
    size = tpr.shape[0]
    for i in range(1,size):
        x0 = fpr[i-1]
        x = fpr[i]
        y0 = tpr[i-1]
        y = tpr[i]
        res += (x - x0) * y0 #площадь прямоугольника
        res += 0.5 * (x - x0) * (y - y0) #площадь треугольника в случае если y возрос
    return res


def plotROC(tpr, fpr, auc = -1):
    """
    :param tpr: True Positive Rate (sensitivite, чувствительность) в порядке возрастания
    :param fpr: False Positive Rate (specificity, специфичность) в порядке возрастания
    :return: None
    """
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.suptitle("ROC")
    if auc > 0:
        plt.text(0.8, 0.0, "AUC = " + "%.3f" % auc)
    plt.show()
