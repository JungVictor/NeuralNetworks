import warnings
import matplotlib.pyplot as plt
from refiner import *
from config import *

warnings.filterwarnings("ignore")  # just ignore the warning messages (who cares anyway?)

print("DATA PREPARATION")

# Reading the file
wines = pd.read_csv("winemag-data-130k-v2.csv")
wines.drop_duplicates('description', inplace=True)
wines.reset_index(drop=True, inplace=True)
vintage = title_to_vintage(wines['title'])

# Dropping the index columns, designation, region_2, taster name and twitter, title and winery
wines.drop(wines.columns[[0, 3, 8, 9, 10, 11, 13]], axis=1, inplace=True)
if drop_description:
    wines.drop(wines.columns[[1]], axis=1, inplace=True)

# Replacing region by the province if there is no region defined
new_region = []

for i in range(len(wines["region_1"])):
    if pd.isna(wines["region_1"][i]):
        new_region.append(wines["province"][i])
    else:
        new_region.append(wines["region_1"][i])

wines["region_1"] = new_region
wines["vintage"] = vintage

# Dropping all lines where there's an undefined value
wines.dropna(inplace=True)
wines.reset_index(drop=True, inplace=True)


# Filtering the variety
wines['variety'] = wines['variety'].str.lower()

# Changing the name of the variety
filtered_name = ['red blend', 'portuguese red', 'white blend', 'sparkling blend', 'champagne blend',
                 'portuguese white', 'rose', 'bordeaux-style red blend', 'rhone-style red blend',
                 'bordeaux-style white blend', 'alsace white blend', 'austrian red blend',
                 'austrian white blend', 'cabernet blend', 'malbec blend', 'portuguese rose',
                 'portuguese sparkling', 'provence red blend', 'provence white blend',
                 'rhone-style white blend', 'tempranillo blend', 'grenache blend',
                 'meritage']

name_pairs = [('spatburgunder', 'pinot noir'), ('garnacha', 'grenache'), ('pinot nero', 'pinot noir'),
              ('alvarinho', 'albarino'), ('assyrtico', 'assyrtiko'), ('black muscat', 'muscat hamburg'),
              ('kekfrankos', 'blaufrankisch'), ('garnacha blanca', 'grenache blanc'),
              ('garnacha tintorera', 'alicante bouschet'), ('sangiovese grosso', 'sangiovese')]

wines['variety'] = wines['variety'].apply(lambda row: correct_grape_names(row))
for start, end in name_pairs:
    wines['variety'] = wines['variety'].replace(start, end)

# Drop all variety who has less than 200 reviews
wines = wines.groupby('variety').filter(lambda x: len(x) > 200)

# DATE FOR PLOT AFTER REFINING
country_count_after = {}
for e in wines["country"]:
    if e not in country_count_after:
        country_count_after[e] = 1
    else:
        country_count_after[e] += 1

price_distribution = {}
for p in wines['price']:
    if p not in price_distribution:
        price_distribution[p] = 1
    else:
        price_distribution[p] += 1

price_distribution = sorted(price_distribution.items(), key=lambda k: k[0])
price_distribution_keys = [x[0] for x in price_distribution]
price_distribution_values = [x[1] for x in price_distribution]

# Changing price to price range
wines["price"], range_number = from_price_to_range(wines["price"], price_range)

# SAVING RAW DATASET
# Renaming columns and saving to csv
wines.to_csv(output_dir + "raw_" + data_filename, sep=csv_separator, encoding='utf-8', index=False)

plt.figure(1)
# we prepare the ploting
plt.ylabel('Number of wines')   # labeling y-axis
plt.xlabel('Price (USD)')              # labeling x-axis
# this line is only here to rotate the labels on the x-axis if there is not enough space
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.bar(price_distribution_keys, price_distribution_values, width=0.8, bottom=None, align='center', data=None, color='green')


# Hashing data
wines.reset_index(drop=True, inplace=True)
wines["country"], country_table = string_hash(wines["country"])
wines["region_1"], region_table = intelligent_region_province_hash(wines["region_1"], wines["country"], 370)
wines["province"], province_table = intelligent_region_province_hash(wines["province"], wines["country"], 50)
wines["variety"], variety_table = string_hash(wines["variety"])

# Useful words for the description hash
useful_words = ['fruit', 'tannins', 'cherry', 'ripe', 'black', 'spice', 'red', 'oak', 'berry', 'dry', 'plum', 'apple',
                'blackberry', 'soft', 'white', 'crisp', 'sweet', 'citrus', 'Cabernet', 'vanilla', 'dark', 'light',
                'bright', 'pepper', 'juicy', 'raspberry', 'green', 'firm', 'peach', 'lemon', 'chocolate', 'dried',
                'balanced', 'Sauvignon', 'Pinot', 'smooth', 'licorice', 'herb', 'earth', 'tannic']
if not drop_description:
    wines["description"], words_table, r_words_table = description_hash(wines["description"], useful_words)

# INFORMATIONS
print("Number of countries : {}".format(len(country_table)))
print("Number of regions : {}".format(len(region_table)))
print("Number of province : {}".format(len(province_table)))
print("Number of variety : {}".format(len(variety_table)))

# SAVING HASH TABLES
dict_to_csv(country_table, output_dir, country_filename)
dict_to_csv(region_table, output_dir, region_filename)
dict_to_csv(province_table, output_dir, province_filename)
dict_to_csv(variety_table, output_dir, variety_filename)

# Renaming columns and saving to csv
wines.rename(index=str, columns={"region_1": "region", "price": "price_range"}, inplace=True)
wines.to_csv(output_dir + data_filename, sep=csv_separator, encoding='utf-8', index=False)

##########
# PLOTTING
##########

####################
# REVIEW PER COUNTRY
####################
top_countries_after = sorted(country_count_after.items(), key=lambda k: k[1], reverse=True)[:10]

data_after_values = [x[1] for x in top_countries_after]
data_after_keys = [x[0] for x in top_countries_after]

plt.figure(2)
plt.subplot(211)
# we prepare the ploting
plt.ylabel('Number of wines per country')   # labeling y-axis
plt.xlabel('Top 10 countries')              # labeling x-axis
# this line is only here to rotate the labels on the x-axis if there is not enough space
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.bar(data_after_keys, data_after_values, width=0.8, bottom=None, align='center', data=None, color='red')

####################
# PRICE DISTRIBUTION
####################

price_distribution = []
for e in range(len(price_range)+1):
    price_distribution.append(0)

for wine_price in wines['price_range']:
    price_distribution[wine_price] += 1

keys = []
for i in range(len(price_range) + 1):
    if i == 0:
        key = '[0, {}['.format(price_range[0])
    elif i == len(price_range):
        key = '{}+'.format(price_range[i-1])
    else:
        key = '[{}, {}['.format(price_range[i-1], price_range[i])
    keys.append(key)

plt.subplot(212)
# we prepare the ploting
plt.ylabel('Number of wines')   # labeling y-axis
plt.xlabel('Price')              # labeling x-axis
# this line is only here to rotate the labels on the x-axis if there is not enough space
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.bar(keys, price_distribution, width=0.8, bottom=None, align='center', data=None, color='green')

plt.show()
