from config import *
from refiner import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")  # just ignore the warning messages

print("REGION SELECTOR")

#####################
price = 50
points = 87
variety = "Red Blend"

# Testing the network on the 2018 reviews ? (might take some time)
testing_2018 = True

# Too much data from US that interfere with the choice of region
# False, False = All the data
# True, False = Only non-US data
# False, True = Only US data
# True, True = Only non-US data
not_US = False
only_US = False
#####################

# Reading the main dataset
wines = pd.read_csv(output_dir + data_filename, sep=csv_separator)

# Vintage is not important
wines.drop(wines.columns[[-1]], axis=1, inplace=True)

# Reading the hash table
country_table = pd.read_csv(output_dir + country_filename, sep=csv_separator)
region_table = pd.read_csv(output_dir + region_filename, sep=csv_separator)
variety_table = pd.read_csv(output_dir + variety_filename, sep=csv_separator)
province_table = pd.read_csv(output_dir + province_filename, sep=csv_separator)

# Translating the DataFrame to dict
country_table, country_table_r = df_to_dict(country_table)
region_table, region_table_r = df_to_dict(region_table)
variety_table, variety_table_r = df_to_dict(variety_table)
province_table, province_table_r = df_to_dict(province_table)

# Translate from string to int, so the network can understand
price = from_price_to_range([price], price_range)[0][0]
variety = variety_table_r[variety]

only_US = only_US and not not_US

if not_US:
    for i in range(len(wines['country'])):
        if wines['country'][i] == country_table_r['US']:
            wines['country'][i] = np.nan

    wines.dropna(inplace=True)
    wines.reset_index(drop=True, inplace=True)

elif only_US:
    for i in range(len(wines['country'])):
        if wines['country'][i] != country_table_r['US']:
            wines['country'][i] = np.nan

    wines.dropna(inplace=True)
    wines.reset_index(drop=True, inplace=True)

# y is the data we're studying.
y = wines["province"]
# X is the data that we'll use to make correlation with y
X = wines.drop(wines.columns[[0, 3, 4]], axis=1)

# Creating the actual sets we'll use to train the neural network
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99, random_state=200)

######################
# CREATING THE NETWORK
######################
network_shape = (10, 10, 10,)  # 3 hidden layout of size 10

# All tested networks
neighbors = KNeighborsClassifier(15)
gauss = GaussianProcessClassifier()
bayes = GaussianNB()
mlp = MLPClassifier()
svc = SVC(kernel='linear', C=0.025)
decision_tree = DecisionTreeClassifier()
forest = RandomForestClassifier(n_estimators=100, min_samples_split=4)

# Selecting one network for the exercise
neural_network_province = neighbors

# TRAINING THE NETWORK
print("\nBEGINNING OF THE TRAINING...", end='\t')
neural_network_province.fit(X_train, y_train)
print("END OF THE TRAINING")

# Score of the neural network in correctly predicting
score = neural_network_province.score(X_test, y_test)
print("Score of the classifier (province) : {:.1%}".format(score))

#########################################
# Testing the network on the 2018 reviews
#########################################
X = pd.read_csv('test/review.csv', sep=csv_separator)
if not_US:
    for i in range(len(X['country'])):
        if X['country'][i] == 'US':
            X['country'][i] = np.nan

elif only_US:
    for i in range(len(X['country'])):
        if X['country'][i] != 'US':
            X['country'][i] = np.nan

X.dropna(inplace=True)
X.reset_index(inplace=True, drop=True)
correct = 0
count = 0
answers = []
unique_answers = []

if testing_2018:
    # Asking the network to answer the problem
    for i in range(len(X)):
        if X['variety'][i] in variety_table_r and X['province'][i] in province_table_r:
            count += 1
            pts = X['points'][i]
            vrt = variety_table_r[X['variety'][i]]
            prc = from_price_to_range([X['price'][i]], price_range)[0][0]
            vnt = X['vintage'][i]
            wine = [[pts, prc, vrt]]
            answer = neural_network_province.predict(wine)[0]
            expected = province_table_r[X['province'][i]]

            answers.append(answer)
            if answer not in unique_answers:
                unique_answers.append(answer)

            if answer == expected:
                correct += 1

    print('Score on the 2018 data (province) : {}/{} ({:.1%})'.format(correct, count, correct / count))

    # SORTING DATA BY COUNTRY IDENTIFIED AS SOLUTION
    print("\nSORTING DATA BY PROVINCE...", end='\t')
    province_data = {}

    for i in range(len(wines)):
        wine = [wines['province'][i], wines['points'][i], wines['price_range'][i], wines['variety'][i],
                wines['region'][i]]
        if wine[0] in unique_answers:
            if wine[0] not in province_data:
                province_data[wine[0]] = {'points': [], 'price': [], 'variety': [], 'region': []}

            province_data[wine[0]]['points'].append(wine[1])
            province_data[wine[0]]['price'].append(wine[2])
            province_data[wine[0]]['variety'].append(wine[3])
            province_data[wine[0]]['region'].append(wine[4])

    print("DATA SORTED BY PROVINCE")

    # CREATE A NETWORK FOR EACH COUNTRY IDENTIFIED AS SOLUTION
    country_networks = {}

    for country in unique_answers:
        network = KNeighborsClassifier(15)

        data = province_data[country]
        data = pd.DataFrame(data)

        y_country = data['region']
        X_country = data.drop(data.columns[[-1]], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_country, y_country, train_size=0.99, random_state=200)
        network.fit(X_train, y_train)
        country_networks[country] = network

    # Solving the problem
    count = 0
    correct = 0
    for i in range(len(X)):
        if X['variety'][i] in variety_table_r and X['region_1'][i] in region_table_r and X['province'][i] in province_table_r and province_table_r[X['province'][i]] in country_networks:
            neural_network = country_networks[province_table_r[X['province'][i]]]
            count += 1
            pts = X['points'][i]
            vrt = variety_table_r[X['variety'][i]]
            prc = from_price_to_range([X['price'][i]], price_range)[0][0]
            vnt = X['vintage'][i]
            wine = [[pts, prc, vrt]]
            answer = neural_network.predict(wine)[0]
            expected = region_table_r[X['region_1'][i]]

            answers.append(answer)
            if answer not in unique_answers:
                unique_answers.append(answer)

            if answer == expected:
                correct += 1

    print('Score on the 2018 data (region) : {}/{} ({:.1%})'.format(correct, count, correct / count))

######################################
# END TEST 2018
######################################

# Data of the problem
problem = [[points, price, variety]]

# Country guessed by the first neural network
province = neural_network_province.predict(problem)[0]

############################################
# Building the neural network for the region

# Collecting data for the given country
print("\nCOLLECTING DATA FOR {}...".format(province_table[province]), end='\t')
if testing_2018 and country in province_data:
    province_data = province_data[province]
else:
    province_data = {'points': [], 'price': [], 'variety': [], 'region': []}
    for i in range(len(wines)):
        wine = [wines['province'][i], wines['points'][i], wines['price_range'][i], wines['variety'][i],
                wines['region'][i]]
        if wine[0] == province:
            province_data['points'].append(wine[1])
            province_data['price'].append(wine[2])
            province_data['variety'].append(wine[3])
            province_data['region'].append(wine[4])

print("END OF COLLECTION")
data = pd.DataFrame(province_data)

# Creating the neural network
neural_network_region = KNeighborsClassifier(15)

# Creating the training data
y_country = data['region']
X_country = data.drop(data.columns[[-1]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_country, y_country, train_size=0.99, random_state=200)

# Training the neural network
neural_network_region.fit(X_train, y_train)

# Score of the neural network in correctly predicting
score = neural_network_region.score(X_test, y_test)
print("Score of the classifier (region) : {:.1%}".format(score))

# Answer of the neural network
region = neural_network_region.predict(problem)[0]
print('\nANSWER :')
print('Selected region : {}, {}'.format(region_table[region], province_table[province]))
