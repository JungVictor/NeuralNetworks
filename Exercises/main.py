import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")   # just ignore the warning messages (who cares anyway?)

#########################
# DEALING WITH NaN VALUES
#########################

# reading the data. na_values="   " means that 3 spaces must be detected as NaN values by the reader
deaths = pd.read_csv("deathsfrance18991995.csv", na_values="   ")
# we then fill all blank (NaN) values with 0 (integer)
# inplace means that we do the action on the list itself (otherwise, it returns a list with filled values)
deaths.fillna(0, inplace=True)

###############################
# REMOVING COLUMNS FROM DATASET
###############################

# we create the list Age12 where Age12 = Age1 * 10 + Age2 for each elements of the list
deaths["Age12"] = [int(deaths["Age1"][i]) * 10 + deaths["Age2"][i] for i in range(len(deaths))]
# we then drop the columns 1 and 2 (deaths.columns[[1,2]] = ["Age1", "Age12"])
deaths.drop(deaths.columns[[1,2]], axis=1, inplace=True)

#########################
# REARRANGE COLUMNS ORDER
#########################

# we then rearrange the values so that Age12 is the second value on the dataset
cols = deaths.columns.tolist()
# cols = "Year" + "Age12" + rest of the list
# cols = ["Year", "Age12", ...] cols is a list, it's important
cols = [cols[0]] + [cols[-1]] + cols[1:-1]
# here, we properly rearrange
deaths = deaths[cols]

################
# BUILDING DATAS
################

# we want to build a plot with informations from year 1900 and 1990.
years = [1900, 1990]
# for each year in the years array
for y in years:
    number_of_death = []    # we fill this array with integer values. it contains the number of total deaths
    age_of_death = []       # we fill this array with integer values. it contains the age of death
    number_of_death_f = []  # same. it contains the number of female death
    for i in range(len(deaths)):
        if deaths["Year"][i] == y:  # if we're looking at a right year
            number_of_death.append(deaths["Total"][i])      # we add the data to the arrays
            number_of_death_f.append(deaths["Female"][i])
            age_of_death.append(deaths["Age12"][i])

# we build the array that contains the difference between male and female number of death
# when it's positive, it means that there is more death in male population
# when it's negative, it means that there is more death in female population
diff = [number_of_death[i]-(number_of_death_f[i]*2) for i in range(len(number_of_death))]

######################
# PLOTTING THE RESULTS
######################

# we prepare the ploting
plt.ylabel('Number of death')   # labeling y-axis
plt.xlabel('Age of death')      # labeling x-axis
# this line is only here to rotate the labels on the x-axis if there is not enough space
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')

# ploting the number of death (total) in blue, bar style
plt.bar(age_of_death, number_of_death, width=0.8, bottom=None, align='center', data=None, color='blue')
# ploting the number of death (female) in pink, bar style
plt.bar(age_of_death, number_of_death_f, width=0.8, bottom=None, align='center', data=None, color='pink')
# ploting the difference between male and female death
plt.plot(age_of_death, diff, color='black')

# in the end, we obtain a kind of stacking bar plot.

plt.legend()    # show the legend
plt.show()      # show the graphic


########################
# CREATING TRAINING SETS
########################
# y is the data we're studying.
y = deaths["Male"]
# X is the data that we'll use to make correlation with y
X = deaths.drop(deaths.columns[[3]], axis=1)

# Creating the actual sets we'll use to train the neural network (yay!)
# by default, train_size=0.25, which means 1/4th of the dataset will serve for training
# but 0.25 takes sooooooooo much time. Feel free to modify the value (between 1.0 and 0.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)

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
mlp = MLPClassifier(hidden_layer_sizes=network_shape, max_iter=1000)

# TRAINING THE NETWORK

print("BEGINING OF THE TRAINING")
mlp.fit(X_train, y_train)
print("END OF THE TRAINING")

# ANALYZING THE RESULTS

# We try to predict using the networks we trained before
y_pred = mlp.predict(X_test)
# We build the confusion matrix
# https://en.wikipedia.org/wiki/Confusion_matrix
matrix = confusion_matrix(y_test, y_pred)

print(matrix[:10])   # print elements from 0 to 10 ( [0, 10[ interval )

# We realize that is seems like there's only the first column that has actual values !

report = classification_report(y_test, y_pred)
print(report)

# Well... seems like shit.

# Get the score of the classifier.
# It represents at what rate it guess successfully.
score = mlp.score(X_test, y_test)
print("Score of the classifier : {}".format(score))

# The score is awful. It means that our prediction are very bad

#####################################################
# COMPARISON BETWEEN PREDICTED VALUES AND ACTUAL DATA
#####################################################

# We trained our network to make connection between the (year, age, female, total) AND male.
# So we're now plotting to show differences between prediction and actual values on a given year
year = [1900, 1943, 1990]

# BUILDING DATA
for y in year:
    years = []
    ages = []
    female = []
    male = []
    total = []
    for i in range(len(deaths)):
        if deaths["Year"][i] == y:
            years.append(y)
            ages.append(deaths["Age12"][i])
            female.append(deaths["Female"][i])
            male.append(deaths["Male"][i])
            total.append(deaths["Total"][i])

    data = pd.DataFrame(data=years, columns=["Year"])
    data["Age12"] = ages
    data["Female"] = female
    data["Total"] = total

    # Creating prediction ...
    prediction = mlp.predict(data)

    plt.title(y)
    plt.ylabel('Number of death')   # labeling y-axis
    plt.xlabel('Age of death')      # labeling x-axis
    plt.plot(ages, male, color='blue', label="Actual")
    plt.plot(ages, prediction, color='red', label="Prediction")

    plt.legend()
    plt.show()

# Results depends on the run. Sometimes results will be good, sometimes bad without changing a single line of code...
# Try to run it multiples times, just to see.
