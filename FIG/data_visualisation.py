from config import *
from PRE.refiner import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # just ignore the warning messages

print("DATA VISUALISATION")

# Reading the main dataset
wines = pd.read_csv(output_dir + data_filename, sep=csv_separator)
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

###################
# PLOTTING
###################

correlation_matrix = wines.corr()
print("CORRELATION MATRIX :\n{}".format(correlation_matrix))

################
# PRICE SELECTOR
################

# AVERAGE PRICE RANGE PER COUNTRY
average_price_range = []
top_countries = [1, 2, 6, 5, 7, 0, 4, 8, 9, 3]
country_average = wines.groupby(['country']).mean()['price_range']
for i in range(len(top_countries)):
    average_price_range.append(country_average[top_countries[i]])
    top_countries[i] = country_table[top_countries[i]]

plt.figure(1)
plt.title('Average price range per country')
plt.xlabel('Country')
plt.ylabel('Average price range')
plt.plot(top_countries, average_price_range)


# AVERAGE POINTS PER PRICE RANGE
average_points_per_interval = wines.groupby('price_range').mean()['points']

plt.figure(2)
plt.title('Average points per price range')
plt.xlabel('Price range')
plt.ylabel('Points')
plt.plot(range(len(price_range) + 1), average_points_per_interval)

# AVERAGE PRICE PER VARIETY
average_price_variety = wines.groupby('variety').mean()['price_range'].sort_values()
plt.figure(3)
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title('Average price range per variety')
plt.xlabel('Variety')
plt.ylabel('Price range')
plt.plot([variety_table[v] for v in average_price_variety.keys()], average_price_variety)


##################
# VARIETY SELECTOR
##################

# MOST USED VARIETY PER COUNTRY
print("MOST USED VARIETY PER COUNTRY")
top_countries = [1, 2, 6, 5, 7, 0, 4, 8, 9, 3]
top_3_variety = [[], [], []]
most_used_variety = wines.groupby(['country', 'variety']).count()['points']
for i in range(len(top_countries)):
    sorted = most_used_variety[top_countries[i]].sort_values(ascending=False)
    count = 0
    for key in sorted.keys():
        top_3_variety[count].append(variety_table[key])
        count += 1
        if count == 3:
            break
    top_countries[i] = country_table[top_countries[i]]

for i in range(len(top_countries)):
    print("{} : ({}, {}, {})".format(top_countries[i], top_3_variety[0][i], top_3_variety[1][i], top_3_variety[2][i]))


# AVERAGE POINTS PER VARIETY
average_points_per_variety = wines.groupby(['variety']).mean()['points'].sort_values()
plt.figure(7)
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title('Average points per variety')
plt.xlabel('Variety')
plt.ylabel('Points')
plt.plot([variety_table[v] for v in average_points_per_variety.keys()], average_points_per_variety)


#################
# REGION SELECTOR
#################

# NUMBER OF REGION AND PROVINCE PER COUNTRY
number_of_region = wines.groupby(['country', 'region']).mean()['price_range']
number_of_province = wines.groupby(['country', 'province']).mean()['price_range']
plt.figure(4)
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title('Number of region and province per country')
plt.xlabel('Country')
plt.ylabel('Number')

top_countries = [1, 2, 6, 5, 7, 0, 4, 8, 9, 3]
top_countries_region = [len(number_of_region[c]) for c in top_countries]
top_countries_province = [len(number_of_province[c]) for c in top_countries]

plt.plot([country_table[c] for c in top_countries], top_countries_region, label='Number of region')
plt.plot([country_table[c] for c in top_countries], top_countries_province, label='Number of province')
plt.legend()

# AVERAGE PRICE RANGE PER PROVINCE
countries = ['France', 'US']
country = country_table_r[countries[0]]
plt.figure(5)
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title('Average price range per province')
plt.xlabel('Province')
plt.ylabel('Price range')
plt.plot([province_table[p] for p in number_of_province[country].keys()], number_of_province[country], label=countries[0])
country = country_table_r[countries[1]]
plt.plot([province_table[p] for p in number_of_province[country].keys()], number_of_province[country], label=countries[1])
plt.legend()

# AVERAGE POINTS PER PROVINCE
points_per_province = wines.groupby(['country', 'province']).mean()['points']
countries = ['France', 'US']
country = country_table_r[countries[0]]
plt.figure(6)
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title('Average points per province')
plt.xlabel('Province')
plt.ylabel('Points')
plt.plot([province_table[p] for p in number_of_province[country].keys()], points_per_province[country], label=countries[0])
country = country_table_r[countries[1]]
plt.plot([province_table[p] for p in number_of_province[country].keys()], points_per_province[country], label=countries[1])
plt.legend()

#############

plt.show()
