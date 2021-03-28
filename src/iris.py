from sklearn import datasets

def getDatasetForClasses(class0, class1):
    iris = datasets.load_iris()
    iris0 = iris.data[iris.target==class0]
    iris1 = iris.data[iris.target==class1]
    return iris0, iris1
