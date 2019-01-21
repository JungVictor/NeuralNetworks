######################
#  DATA PREPARATION  #
######################
# Name of the output dataset
data_filename = 'dataset.csv'
country_filename = 'country_table.csv'
price_filename = 'price_table.csv'
region_filename = 'region_table.csv'
province_filename = 'province_table.csv'
variety_filename = 'variety_table.csv'

# Directory of the outputs csv
output_dir = 'output/'

# Separator used for the csv format
csv_separator = ';'

# Drop the description column ?
drop_description = True

# Price range
uniform = [14, 19, 24, 30, 40, 55, 100]
curve = [10, 15, 20, 27, 40, 55, 100]
realist = [10, 20, 50, 80, 100, 150, 200]

price_range = realist
#####################
