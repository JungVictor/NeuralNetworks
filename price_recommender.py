import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")   # just ignore the warning messages (who cares anyway?)


wines = pd.read_csv("output.csv", sep=';')

# y is the data we're studying.
y = wines["price_range"]
# X is the data that we'll use to make correlation with y
X = wines.drop(wines.columns[[1]], axis=1)

# Creating the actual sets we'll use to train the neural network (yay!)
# by default, train_size=0.25, which means 1/4th of the dataset will serve for training
# but 0.25 takes sooooooooo much time. Feel free to modify the value (between 1.0 and 0.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99)

# we fit the scaler with X_train
scaler = StandardScaler().fit(X_train)

# then we scale training and test sets
scaler.transform(X_train)
scaler.transform(X_test)

######################
# CREATING THE NETWORK
######################

# Multi-layer Perceptron, not My Little Pony
network_shape = (100, 100, 100)    # 3 hidden layout of size 100
neighbors = KNeighborsClassifier(10)


# TRAINING THE NETWORK
print("BEGINNING OF THE TRAINING")
neighbors.fit(X_train, y_train)
print("END OF THE TRAINING")

score = neighbors.score(X_test, y_test)
print("Score of the classifier : {}".format(score))