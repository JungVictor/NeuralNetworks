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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")   # just ignore the warning messages (who cares anyway?)

#####################
country = 'US'
points = 87
province = 'California'
region = 'Napa'
variety = 'Pinot Gris'
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
variety = variety_table_r[variety]
province = province_table_r[province]

# y is the data we're studying.
y = wines["price_range"]
# X is the data that we'll use to make correlation with y
X = wines.drop(wines.columns[[2]], axis=1)

# Creating the actual sets we'll use to train the neural network (yay!)
# by default, train_size=0.25, which means 1/4th of the dataset will serve for training
# but 0.25 takes sooooooooo much time. Feel free to modify the value (between 1.0 and 0.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99, random_state=165)

######################
# CREATING THE NETWORK
######################

# Multi-layer Perceptron, not My Little Pony
network_shape = (100, 100, 100)    # 3 hidden layout of size 10
neighbors = KNeighborsClassifier(1)
gauss = GaussianProcessClassifier()
bayes = GaussianNB()
mlp = MLPClassifier(hidden_layer_sizes=network_shape)
svc = SVC(kernel='linear', C=0.025)
decision_tree = DecisionTreeClassifier()
forest = RandomForestClassifier(n_estimators=100, min_samples_split=4)

neural_network = neighbors

# TRAINING THE NETWORK
print("BEGINNING OF THE TRAINING")
for i in range(150):
    neural_network.fit(X_train, y_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99, random_state=i)
    score = neural_network.score(X_test, y_test)
    if score > 0.5:
        break
print("END OF THE TRAINING")


score = neural_network.score(X_test, y_test)

difference = 0
actual = neural_network.predict(X_test)
expected = [e for e in y_test]
count = 0
for i in range(len(actual)):
    if actual[i] != expected[i]:
        difference += abs(actual[i] - expected[i])
        count += 1

difference = difference / count / (len(price_range)+1) * 100
print("Score of the classifier : {}%".format(score*100))
print("Average difference between expected and actual : {}%".format(difference))

# Asking the network to answer the problem
answers = []

X = pd.read_csv('test/review.csv', sep=csv_separator)
X.dropna(inplace=True)
X.reset_index(inplace=True, drop=True)
correct = 0
for i in range(len(X)):
    if X['country'][i] in country_table_r and X['province'][i] in province_table_r and X['region_1'][i] in region_table_r and X['variety'][i] in variety_table_r:
        ctr = country_table_r[X['country'][i]]
        pts = X['points'][i]
        prv = province_table_r[X['province'][i]]
        rgn = region_table_r[X['region_1'][i]]
        vrt = variety_table_r[X['variety'][i]]
        vnt = X['vintage'][i]
        wine = [[ctr, pts, prv, rgn, vrt, vnt]]
        answer = index_to_range(neural_network.predict(wine)[0], price_range)
        expected = X['price'][i]
        if answer[0] <= expected < answer[1]:
            answers.append(True)
            correct += 1
        else:
            answers.append(False)


print(correct, '/', len(X), correct/len(X)*100)