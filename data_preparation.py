import warnings
from refiner import *
from config import *

warnings.filterwarnings("ignore")  # just ignore the warning messages (who cares anyway?)

print("DATA PREPARATION")

# Reading the file
wines = pd.read_csv("winemag-data-130k-v2.csv")
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
dropped = wines.dropna()
print("Dropped : {}%".format((len(wines) - len(dropped)) / len(wines) * 100))
wines = dropped
wines.reset_index(drop=True, inplace=True)

# Hashing data
wines["country"], country_table = string_hash(wines["country"])
wines["region_1"], region_table = string_hash(wines["region_1"])
wines["province"], province_table = string_hash(wines["province"])
wines["variety"], variety_table = string_hash(wines["variety"])
wines["price"], range_number = from_price_to_range(wines["price"], price_range)

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
