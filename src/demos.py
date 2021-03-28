import iris
import ROC
import classifiers
import util
import numpy as np
import ellipses
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def printRes(tpr, fpr, thresholds, auc):
    print("tpr = \n" + str(tpr))
    print("fpr = \n" + str(fpr))
    print("thresholds = \n" + str(thresholds))
    print("AUC = " + str(auc))

def ROCdemo(class0, class1):
    iris0, iris1 = iris.getDatasetForClasses(class0, class1)
    classifier = lambda x: x.T[0] - x.T[1] + x.T[2] - x.T[3]
    tpr, fpr, thresholds = util.classifierROC(classifier, iris0, iris1)
    auc = ROC.AUC(tpr, fpr)
    printRes(tpr, fpr, thresholds, auc)
    ROC.plotROC(tpr,fpr, auc)
    #fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    #ROC.plotROC(tpr,fpr)

def FisherClassifierDemo(class0, class1, trainPercent = 0.8):
    iris0, iris1 = iris.getDatasetForClasses(class0, class1)
    n0 = iris0.shape[0]
    n1 = iris1.shape[0]
    n0_train = round(n0 * trainPercent)
    n1_train = round(n1 * trainPercent)
    iris0train = iris0[:n0_train]
    iris1train = iris1[:n1_train]
    iris0test = iris0[n0_train:]
    iris1test = iris1[n1_train:]
    classifierFisher, biasFisher = classifiers.trainFishersLinearDiscriminant(iris0train, iris1train)
    tpr, fpr, thresholds = util.classifierROC(classifierFisher, iris0test, iris1test)
    auc = ROC.AUC(tpr, fpr)
    printRes(tpr, fpr, thresholds, auc)
    ROC.plotROC(tpr,fpr, auc)

def NaiveBayesClassifierDemo(trainPercent = 0.8):
    docs, labels = util.extractDocsAndLabels("..\Files\spam0.csv")
    bagsOfWords, vocab = util.preprocessDocs(docs)
    x_train, y_train, x_test, y_test = util.getDataset(bagsOfWords, labels, trainPercent)
    labels_train = np.zeros((y_train.shape[0],))
    labels_train[y_train == 'ham'] = 0
    labels_train[y_train == 'spam'] = 1
    BNB = util.trainBinaryNaiveBayes(x_train, labels_train, vocab)
    tpr, fpr, thresholds = util.classifierROC(lambda x: BNB(x, vocab), x_test[y_test == 'spam'], x_test[y_test == 'ham'])
    auc = ROC.AUC(tpr, fpr)
    printRes(tpr, fpr, thresholds, auc)
    ROC.plotROC(tpr,fpr, auc)

def BernulliClassifierDemo(trainPercent = 0.8):
    docs, labels = util.extractDocsAndLabels("..\Files\spam0.csv")
    bagsOfWords, vocab = util.preprocessDocs(docs)
    x_train, y_train, x_test, y_test = util.getDataset(bagsOfWords, labels, trainPercent)
    labels_train = np.zeros((y_train.shape[0],))
    labels_train[y_train == 'ham'] = 0
    labels_train[y_train == 'spam'] = 1
    BB = util.trainBinaryBernulli(x_train, labels_train, vocab)
    tpr, fpr, thresholds = util.classifierROC(lambda x: BB(x, vocab), x_test[y_test == 'spam'], x_test[y_test == 'ham'])
    auc = ROC.AUC(tpr, fpr)
    printRes(tpr, fpr, thresholds, auc)
    ROC.plotROC(tpr,fpr, auc)

def SVMDemo(class0, class1, trainPercent=0.8):
    iris0, iris1 = iris.getDatasetForClasses(class0, class1)
    n0 = iris0.shape[0]
    n1 = iris1.shape[0]
    n0_train = round(n0 * trainPercent)
    n1_train = round(n1 * trainPercent)
    iris0train = iris0[:n0_train]
    iris1train = iris1[:n1_train]
    iris0test = iris0[n0_train:]
    iris1test = iris1[n1_train:]
    x_train = np.concatenate((iris0train, iris1train))
    y_train = np.ones((x_train.shape[0],))
    y_train[n0_train:] = -1
    SVM = classifiers.trainSVM(x_train, y_train)

    x_test = np.concatenate((iris0test, iris1test))
    y_test = np.ones((x_test.shape[0],))
    y_test[n0 - n0_train:] = -1
    y_res = SVM(x_test)

    #-1 positive, 1 negative
    P_test = y_test == -1
    N_test = y_test == 1
    P_res = y_res == -1
    N_res = y_res == 1

    TP = np.sum(P_test == P_res)
    TN = np.sum(N_test == N_res)
    FP = np.sum(P_test == N_res)
    FN = np.sum(N_test == P_res)

    acc = (TP + TN) / (TP + TN + FP + FN)
    print("accuracy = " + str(acc))

def PetuninAndMinVolEllipseDemo(n):
    #points = np.array([[0.,0.],[1.,0.],[0.,1.],[0.5,0.]])
    #points = np.array([[0.,0.],[1.,0.],[0.5,0.5]])
    #points = np.random.randn(n,2) * np.array([[2,2]]) + 10
    points = np.hstack((np.random.rand(n,1),np.random.randn(n,1)))
    np.save("..\Files\points", points)
    #points = np.load("..\Files\pointsPeeling.npy")
    A,c1 = ellipses.min_vol_ellipsoid(points)
    #c2, width, height, alpha = ellipses.Petunin_ellipse(points)
    c2, R, coef, rot, alphas = ellipses.Petunin_ellipsoid(points)
    radii = np.max(R) * coef
    print("Min volume ellipse area = " + str(ellipses.ellipse_vol_matr(A)))
    print("Petunin ellipse area = " + str(ellipses.ellipsoid_vol(radii)))

    #ell = Ellipse(c2, width, height, alpha * 180 / np.pi, fill=False, color='g')
    ell = Ellipse(c2, radii[0] * 2, radii[1] * 2, alphas[0] * 180 / np.pi, fill=False, color='g')
    fig, ax = plt.subplots()
    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)

    plt.plot(*ellipses.ellipse_to_plot_matr(A,c1), 'b')
    plt.plot(points[:,0], points[:,1], 'ro')
    plt.grid(color='lightgray',linestyle='--')

    size = 6
    center = 0
    ax.set_xlim(-size + center, size + center)
    ax.set_ylim(-size + center, size + center)

    plt.show()

def PetuninAndMinVolEllipsoidDemo(n, dim):
    points = np.random.randn(n, dim) * 1 + 10
    #points = np.array([[1,2,3,4],[2,3,4,5],[1,4,2,5],[4,2,2,3],[5,4,1,1],[2,2,1,2],[6,5,1,2],[4,3,4,3],[3,3,3,3],[5,4,5,3],[2,3,2,1]])
    A,c1 = ellipses.min_vol_ellipsoid(points)
    c2, R, coef, rot, alphas = ellipses.Petunin_ellipsoid(points)
    radii = np.max(R) * coef
    print("Min volume ellipsoid volume = " + str(ellipses.ellipse_vol_matr(A)))
    print("Petunin ellipsoid volume = " + str(ellipses.ellipsoid_vol(radii)))

def peelingDemo(n):
    points = np.random.randn(n, 2)
    np.save("..\Files\points", points)
    #points = np.load("..\Files\pointsPeeling.npy")
    c, R, coef, rot, alphas = ellipses.Petunin_ellipsoid(points)
    order_conc = ellipses.peeling_concentric_rel_dists(R)
    order_entr = ellipses.peeling_entrywise(points)
    volumes_conc = np.array([ellipses.ellipsoid_vol(R[k] * coef) for k in order_conc[-1:1:-1]])

    volumes_entr = np.zeros((n-2,))
    c_entr = [None] * (n-2)
    R_entr = [None] * (n-2)
    coef_entr = [None] * (n-2)
    rot_entr = [None] * (n-2)
    alphas_entr = [None] * (n-2)
    for k in range(n, 2, -1):
        c_entr[n-k], R_entr[n-k], coef_entr[n-k], rot_entr[n-k], alphas_entr[n-k] = \
            ellipses.Petunin_ellipsoid(points[order_entr[0:k]])
        volumes_entr[n-k] = ellipses.ellipsoid_vol(np.max(R_entr[n-k]) * coef_entr[n-k])

    fig = plt.figure()
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])
    xlim = 4
    ylim = 4
    color_alpha = np.linspace(0., 0.9, n-2)

    ax1.set_title("Концентричний")
    for k,r in enumerate(R[order_conc[2:]]):
        ax1.add_artist(Ellipse(c, r * coef[0] * 2, r * coef[1] * 2, alphas[0] * 180 / np.pi,
                               fill=False, color=(0., 0., 1., color_alpha[k])))
    ax1.plot(points[:,0], points[:,1], 'go')
    ax1.grid(color='lightgray',linestyle='--')
    ax1.set_xlim(-xlim, xlim)
    ax1.set_ylim(-ylim, ylim)

    ax2.set_title("Поелементний")
    for k, (c, R, coef, alphas) in enumerate(zip(c_entr, R_entr, coef_entr, alphas_entr)):
        r = np.max(R)
        ax2.add_artist(Ellipse(c, r * coef[0] * 2, r * coef[1] * 2, alphas[0] * 180 / np.pi,
                               fill=False, color=(1., 0., 0., color_alpha[n-3-k])))
    ax2.plot(points[:,0], points[:,1], 'go')
    ax2.grid(color='lightgray',linestyle='--')
    ax2.set_xlim(-xlim, xlim)
    ax2.set_ylim(-ylim, ylim)

    ax3.set_title("Площі еліпсів")
    ax3.plot(volumes_conc, 'b', label="Концентричний")
    ax3.plot(volumes_entr, 'r', label="Поелементний")
    ax3.grid(color='lightgray',linestyle='--')
    ax3.legend()
    plt.show()

