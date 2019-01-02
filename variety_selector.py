import pandas as pd
from config import *
from refiner import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")   # just ignore the warning messages (who cares anyway?)

#####################
country = 'Spain'
points = 87
price = 15
province = 'Northern Spain'
region = 'Navarra'
#####################

# Reading the main dataset
wines = pd.read_csv(output_dir+data_filename, sep=csv_separator)

# Reading the hash table
country_table = pd.read_csv(output_dir+country_filename, sep=csv_separator)
region_table = pd.read_csv(output_dir+region_filename, sep=csv_separator)
variety_table = pd.read_csv(output_dir+variety_filename, sep=csv_separator)
province_table = pd.read_csv(output_dir+province_filename, sep=csv_separator)

# Translating the DataFrame to dict
country_table, country_table_r = df_to_dict(country_table)
region_table, region_table_r = df_to_dict(region_table)
variety_table, variety_table_r = df_to_dict(variety_table)
province_table, province_table_r = df_to_dict(province_table)

# Translate from string to int, so the network can understand
country = country_table_r[country]
region = region_table_r[region]
price = from_price_to_range([price], price_range)[0][0]
province = province_table_r[province]

# y is the data we're studying.
y = wines["variety"]
# X is the data that we'll use to make correlation with y
X = wines.drop(wines.columns[[5]], axis=1)

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
network_shape = (10, 10, 10)    # 3 hidden layout of size 10
neighbors = KNeighborsClassifier(15)
gauss = GaussianProcessClassifier()
mlp = MLPClassifier(hidden_layer_sizes=network_shape)
svc = SVC(kernel='linear', C=0.025)
decision_tree = DecisionTreeClassifier()
forest = RandomForestClassifier()

neural_network = neighbors


# TRAINING THE NETWORK
print("BEGINNING OF THE TRAINING")
neural_network.fit(X_train, y_train)
print("END OF THE TRAINING")

score = neural_network.score(X_test, y_test)
print("Score of the classifier : {}".format(score))

difference = 0
actual = neural_network.predict(X_test)
expected = [e for e in y_test]
for i in range(len(actual)):
    difference += abs(actual[i] - expected[i])

print(difference / len(actual))

# Asking the network to answer the problem
X = [country, points, price, province, region]
answer = neural_network.predict([X])[0]
print(variety_table[answer])

