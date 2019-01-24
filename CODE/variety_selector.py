from config import *
from PRE.refiner import *
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")  # just ignore the warning messages

print("VARIETY SELECTOR")

#####################
country = 'France'
points = 98
price = 100
province = 'Alsace'
region = 'Alsace'

# Testing the network on the 2018 reviews ?
# Put 'False' if you want a better answer to a unique problem
testing_2018 = True
#####################

# Reading the main dataset
wines = pd.read_csv(output_dir + data_filename, sep=csv_separator)

# Vintage is not important
wines.drop(wines.columns[[-1]], axis=1, inplace=True)

if testing_2018:
    dataframe_2018 = pd.read_csv(testing_dir+'review.csv', sep=csv_separator)

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
country = country_table_r[country]
region = region_table_r[region]
price = from_price_to_range([price], price_range)[0][0]
province = province_table_r[province]

# Selecting only the data from the countries we need
# Array of the countries we need
country_array = []

if testing_2018:
    for i in range(len(dataframe_2018['country'])):
        ctr = dataframe_2018['country'][i]
        if ctr in country_table_r:
            ctr = country_table_r[ctr]
            if ctr not in country_array:
                country_array.append(ctr)

else:
    country_array.append(country)
    # California is about 1/3 of the wine in the data, so we only select wines from California if
    # the province asked is California.

if country_table[country] != 'US':
    wines['country'] = filter_nan_values(wines['country'], country_array)
    wines.dropna(inplace=True)

# y is the data we're studying.
y = wines["variety"]
# X is the data that we'll use to make correlation with y
X = wines.drop(wines.columns[[5]], axis=1)

# Creating the actual sets we'll use to train the neural network
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=0)

# we fit the scaler with X_train
scaler = StandardScaler().fit(X_train)

# then we scale training and test sets
scaler.transform(X_train)
scaler.transform(X_test)

######################
# CREATING THE NETWORK
######################

# All tested network
neighbors = KNeighborsClassifier(10, algorithm='ball_tree', weights='distance')
mlp = MLPClassifier(activation='logistic', solver='adam', learning_rate='adaptive',
                    hidden_layer_sizes=(100, 100,), max_iter=1000)


# Choosing the network we'll use
neural_network = mlp

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
    X = dataframe_2018
    X['variety'] = X['variety'].str.lower()
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
            wine = [[ctr, pts, prc, prv, rgn]]
            answer = int(neural_network.predict(wine)[0])
            expected = variety_table_r[X['variety'][i]]
            if answer == expected:
                answers.append(True)
                correct += 1
            else:
                answers.append(False)

    print('Score on the 2018 data : {}/{} ({:.1%})'.format(correct, len(answers), correct / len(answers)))

######################################
# END TEST 2018
######################################

# Data of the problem
problem = [[country, points, price, province, region]]

# Answer of the neural network
answer = np.floor(neural_network.predict(problem)[0])

print("\nANSWER :")
print("Variety selected : {}".format(variety_table[answer]))
