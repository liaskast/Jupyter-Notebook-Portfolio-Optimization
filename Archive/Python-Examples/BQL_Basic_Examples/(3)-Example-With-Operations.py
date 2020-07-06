# Import the BQL library
import bql
# Import the bqviz plotting library
import bqviz as bqv


# Create the database connection 
bq = bql.Service()

# Define a list of securities for the request universe
universe = ['IBM US Equity', 'AAPL US Equity']

# Define the date range for the request
date_range = bq.func.range('2015-01-01', '2015-12-31')

# Define a data item for the last price field,
# passing the date range and frequency parameters
price = bq.data.px_last(dates=date_range, frq="D")

# Generate the request using the security universe and data item
request = bql.Request(universe, price)

# Execute the request
response = bq.execute(request)

# Convert the response to a DataFrame and
# display the first five rows to check the results
dataframe = response[0].df()
dataframe.head(5)


# Pass a dictionary to the rename() function
# to modify the default column names
dataframe.rename(
    columns={"DATE": "Date", 
    "CURRENCY": "Currency", 
    'PX_LAST(frq=PER.D,dates=RANGE(2015-01-01,2015-12-31))': "Last Price"}, 
    inplace=True)

# Display the first five rows of the DataFrame to check the results
dataframe.head(5)


# Reset the DataFrame index
reset_index_dataframe = dataframe.reset_index()
# Reindex the DataFrame
reindexed_datafame = reset_index_dataframe.pivot('Date', 'ID', 'Last Price')
# Drop N/A values from the DataFrame
cleaned_dataframe = reindexed_datafame.dropna()
# Display the first five rows of the DataFrame to check the results
cleaned_dataframe.head(5)


# Reset the DataFrame index
reset_index_dataframe = dataframe.reset_index()
# Reindex the DataFrame
reindexed_datafame = reset_index_dataframe.pivot('Date', 'ID', 'Last Price')
# Drop N/A values from the DataFrame
cleaned_dataframe = reindexed_datafame.dropna()
# Display the first five rows of the DataFrame to check the results
cleaned_dataframe.head(5)
