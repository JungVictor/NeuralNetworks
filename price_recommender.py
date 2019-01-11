from config import *
from refiner import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")  # just ignore the warning messages

print("PRICE RECOMMENDER")

#####################
country = 'US'
points = 87
province = 'Oregon'
region = 'Oregon'
variety = 'Pinot Noir'

# Testing the network on the 2018 reviews ?
# Put 'False' if you want a better answer to a unique problem
testing_2018 = True
#####################

# Reading the main dataset
wines = pd.read_csv(output_dir + data_filename, sep=csv_separator)
wines.drop(wines.columns[[-1]], axis=1, inplace=True)
if testing_2018:
    dataframe_2018 = pd.read_csv('test/review.csv', sep=csv_separator)

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
variety = variety_table_r[variety.lower()]
province = province_table_r[province]

# Selecting only the data from the countries we need
# Array of the countries we need
country_array = []

if testing_2018:
    for i in range(len(dataframe_2018['country'])):
        ctr = dataframe_2018['country'][i]
        vrt = dataframe_2018['variety'][i]
        if ctr in country_table_r:
            ctr = country_table_r[ctr]
            if ctr not in country_array:
                country_array.append(ctr)

else:
    country_array.append(country)

if True:
    wines['country'] = filter_nan_values(wines['country'], country_array)
    wines.dropna(inplace=True)

# y is the data we're studying.
y = wines["price_range"]
# X is the data that we'll use to make correlation with y
X = wines.drop(wines.columns[[2]], axis=1)

# Creating the actual sets we'll use to train the neural network
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99, random_state=0)

######################
# CREATING THE NETWORK
######################
network_shape = (30, 20, 20,)  # 3 hidden layout of size 30, 20, 20

# All tested networks
neighbors = KNeighborsRegressor(10, algorithm='ball_tree', weights='distance')
gauss = GaussianProcessClassifier()
bayes = GaussianNB()
mlp = MLPClassifier(hidden_layer_sizes=network_shape, activation='tanh')
svc = SVC(kernel='linear', C=0.025)
decision_tree = DecisionTreeClassifier()
forest = RandomForestClassifier(n_estimators=100, min_samples_split=4)

# Selecting one network for the exercise
neural_network = neighbors

# TRAINING THE NETWORK
print("\nBEGINNING OF THE TRAINING...", end='\t')
neural_network.fit(X_train, y_train)
print("END OF THE TRAINING")

# Difference in price range from actual to expected
difference = 0
actual = neural_network.predict(X_test)
expected = [e for e in y_test]
count = 0
for i in range(len(actual)):
    if actual[i] != expected[i]:
        difference += abs(actual[i] - expected[i])
        count += 1

difference = difference / count

# Score of the neural network in correctly predicting
score = neural_network.score(X_test, y_test)
print("Score of the classifier : {:.1%}".format(score))
print("Average price difference between expected and actual : {:.2%} ({:.2} range)".format(difference/(len(price_range)+1), difference))


#########################################
# Testing the network on the 2018 reviews
#########################################
if testing_2018:
    X = dataframe_2018
    X.dropna(inplace=True)
    X.reset_index(inplace=True, drop=True)
    X['variety'] = X['variety'].str.lower()
    correct = 0
    count = 0
    expected_answers = {}
    actual_answers = {}
    errors = {}
    for i in range(len(price_range) + 1):
        expected_answers[i] = 0
        errors[i] = 0
        actual_answers[i] = 0

    # Asking the network to answer the problem
    for i in range(len(X)):
        if X['country'][i] in country_table_r and X['province'][i] in province_table_r and X['region_1'][i] in region_table_r and X['variety'][i] in variety_table_r:
            count += 1
            ctr = country_table_r[X['country'][i]]
            pts = X['points'][i]
            prv = province_table_r[X['province'][i]]
            rgn = region_table_r[X['region_1'][i]]
            vrt = variety_table_r[X['variety'][i]]
            vnt = X['vintage'][i]
            wine = [[ctr, pts, prv, rgn, vrt]]
            answer = int(neural_network.predict(wine)[0])
            expected = from_price_to_range([X['price'][i]], price_range)[0][0]

            expected_answers[expected] += 1
            actual_answers[answer] += 1

            if answer == expected:
                correct += 1
            else:
                errors[expected] += 1

    expected_values = list(expected_answers.values())
    actual_values = list(actual_answers.values())
    errors = list(errors.values())

    keys = []
    for i in range(len(price_range) + 1):
        if i == 0:
            key = '[0, {}['.format(price_range[0])
        elif i == len(price_range):
            key = '{}+'.format(price_range[i-1])
        else:
            key = '[{}, {}['.format(price_range[i-1], price_range[i])
        keys.append(key)

    plt.plot(keys, expected_values, color='blue', label='Expected')
    plt.plot(keys, actual_values, color='green', label='Actual')
    plt.plot(keys, errors, color='red', label='Errors')
    plt.legend()
    plt.show()

    print('Score on the 2018 data : {}/{} ({:.1%})'.format(correct, count, correct / count))

######################################
# END TEST 2018
######################################

# Data of the problem
problem = [[country, points, province, region, variety]]

# Answer of the neural network
brute = neural_network.predict(problem)[0]
answer = index_to_range(int(brute), price_range)

print('\nANSWER :')
if answer[0] == price_range[-1]:
    print('Price range recommended : more than ${}'.format(answer[0]))
else:
    print('Price range recommended : from ${} to ${} ({})'.format(answer[0], answer[1], (answer[1] - answer[0]) * (brute - int(brute)) + answer[0]))
