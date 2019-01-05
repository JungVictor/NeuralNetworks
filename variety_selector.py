from config import *
from refiner import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")   # just ignore the warning messages

print("VARIETY SELECTOR")

#####################
country = 'France'
points = 98
price = 220
province = "France Other"
region = 'Alsace'
vintage = 2017

# Testing the network on the 2018 reviews ?
testing_2018 = True
#####################

# Reading the main dataset
wines = pd.read_csv(output_dir+data_filename, sep=csv_separator)
# wines.drop(wines.columns[[6]], axis=1, inplace=True)

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

# Creating the actual sets we'll use to train the neural network
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99, random_state=0)

######################
# CREATING THE NETWORK
######################

# All tested network
network_shape = (10, 10, 10)    # 3 hidden layout of size 10
neighbors = KNeighborsClassifier(15, algorithm='ball_tree')
gauss = GaussianProcessClassifier()
mlp = MLPClassifier(hidden_layer_sizes=network_shape)
svc = SVC(kernel='linear', C=0.025)
decision_tree = DecisionTreeClassifier()
forest = RandomForestClassifier()

# Choosing the network we'll use
neural_network = neighbors

# TRAINING THE NETWORK
print("\nBEGINNING OF THE TRAINING...", end='\t')
neural_network.fit(X_train, y_train)
print("END OF THE TRAINING")

score = neural_network.score(X_test, y_test)
print("Score of the classifier : {:.1%}".format(score))

#########################################
# Testing the network on the 2018 reviews
#########################################

if testing_2018:
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
            prc = from_price_to_range([X['price'][i]], price_range)[0][0]
            vnt = X['vintage'][i]
            wine = [[ctr, pts, prc, prv, rgn, vnt]]
            answer = neural_network.predict(wine)[0]
            expected = variety_table_r[X['variety'][i]]
            if answer == expected:
                answers.append(True)
                correct += 1
            else:
                answers.append(False)

    print('Score on the 2018 data : {}/{} ({:.1%})'.format(correct,  len(answers), correct/len(answers)))

######################################
# END TEST 2018
######################################

# Data of the problem
problem = [[country, points, price, province, region, vintage]]

# Answer of the neural network
answer = neural_network.predict(problem)[0]

print("\nANSWER :")
print("Variety selected : {}".format(variety_table[answer]))
