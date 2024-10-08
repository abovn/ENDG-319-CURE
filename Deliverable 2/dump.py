import sklearn.datasets
cancer = sklearn.datasets.load_breast_cancer()
cancer.keys()

print(cancer.feature_names)

print(cancer.target_names)

print(cancer.DESCR)

