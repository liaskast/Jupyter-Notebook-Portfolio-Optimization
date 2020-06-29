#Example 1: Requesting a Single Data Item
#=============================================================================
'''
bq.data: Defines the data item to include in the request.
  `bq.data. + field(arguments)`
bql.Request(): Generates the request.
  `bql.Request(security or universe, data items)`
bq.execute(): Executes the request.
  `bq.execute(request)`
'''
  
# Import the BQL library
import bql

# Instantiate an object to interface with the BQL service
bq = bql.Service()

# Define a variable for the security 
security = "AAPL US Equity"

# Define a data item for the last price field
# In this example, we're not passing any arguments
last = bq.data.px_last()

# Generate the request using the sercurity variable and data item
request =  bql.Request(security, last)

# Execute the request
response = bq.execute(request)

# Display the response in a DataFrame
response[0].df()
  

#Example 2: Requesting Multiple Data Items
#===========================================================================
# Import the BQL library
import bql

# Instantiate an object to interface with the BQL service
bq = bql.Service()

# Define a variable for the security 
security = "AAPL US Equity"

# Define data items for the pricing fields
last = bq.data.px_last()
high = bq.data.px_high()
low = bq.data.px_low()

# Generate the request using the security and a list of the data items
request = bql.Request("AAPL US Equity", [last, high, low])

# Execute the request
response = bq.execute(request)

# Display the response in a DataFrame
# Use the combined_df function to display the three returned values in a single DataFrame 
bql.combined_df(response)



#Example 3: Requesting Time Series and Point-In-Time Data
#============================================================================
# Import the BQL library
import bql

# Instantiate an object to interface with the BQL service
bq = bql.Service()

# Define the date range for the request
date_range = bq.func.range('2017-06-05','2017-06-09')

# Define data items for the pricing fields
# Pass the defined date range
last = bq.data.px_last(dates=date_range)
high = bq.data.px_high(dates=date_range)
low = bq.data.px_low(dates=date_range)

# Generate the request using the security ticker and a list of the data items
request = bql.Request("AAPL US Equity", [last, high, low])

# Execute the request
response = bq.execute(request)

# Display the response in a data frame
# Use the combined_df function to display 
# the three returned values in a single data frame 
bql.combined_df(response)



#Example 4: Requesting Data for Multiple Securities
#============================================================================
# Import the BQL library
import bql

# Instantiate an object to interface with the BQL service
bq = bql.Service()

# Define a universe of the members of the INDU Index
indu = bq.univ.members("INDU Index")

# Define the date range for the request
date_range = bq.func.range('2017-06-05','2017-06-09')

# Define data items for the pricing fields
# Pass the defined date range
last = bq.data.px_last(dates=date_range)
high = bq.data.px_high(dates=date_range)
low = bq.data.px_low(dates=date_range)

# Generate the request using the security universe and a list of the data items
request = bql.Request(indu, [last, high, low])

# Execute the request
response = bq.execute(request)

# Display the response in a DataFrame
# Use the combined_df function to display 
# the three returned values in a single DataFrame
# To verify the output, use tail(3) to show the last three rows of the response
bql.combined_df(response).tail(3)
