import pandas as pd
import warnings
from refiner import *
warnings.filterwarnings("ignore")   # just ignore the warning messages (who cares anyway?)

#####################
#       INIT        #
#####################
# Name of the output dataset
filename = 'output.csv'

# Drop the description column ?
drop_description = True

# Price range
price_range = [10, 15, 20, 25, 30, 40, 55, 100]
#####################

# Reading the file
wines = pd.read_csv("winemag-data-130k-v2.csv")

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

# Dropping all lines where there's an undefined value
dropped = wines.dropna()
print("Dropped : {}%".format((len(wines) - len(dropped))/ len(wines)*100))
wines = dropped

# Hashing data
wines["country"], country_table = string_hash(wines["country"])
wines["region_1"], region_table = string_hash(wines["region_1"])
wines["province"], province_table = string_hash(wines["province"])
wines["variety"], variety_table = string_hash(wines["variety"])
wines["price"], range_number = from_price_to_range(wines["price"], price_range)

# Useful words for the description hash
useful_words = ['fruit', 'tannins', 'cherry', 'ripe', 'black', 'spice', 'red', 'oak', 'berry', 'dry', 'plum', 'apple', 'blackberry', 'soft', 'white', 'crisp', 'sweet', 'citrus', 'Cabernet', 'vanilla', 'dark', 'light', 'bright', 'pepper', 'juicy', 'raspberry', 'green', 'firm', 'peach', 'lemon', 'chocolate', 'dried', 'balanced', 'Sauvignon', 'Pinot', 'smooth', 'licorice', 'herb', 'earth', 'tannic']
if not drop_description:
    wines["description"], words_table, r_words_table = description_hash(wines["description"], useful_words)

# INFORMATIONS
print("Number of countries : {}".format(len(country_table)))
print("Number of regions : {}".format(len(region_table)))
print("Number of province : {}".format(len(province_table)))
print("Number of variety : {}".format(len(variety_table)))

# Renaming columns and saving to csv
wines.rename(index=str, columns={"region_1": "region", "price": "price_range"}, inplace=True)
wines.to_csv(filename, sep=';', encoding='utf-8', index=False)
