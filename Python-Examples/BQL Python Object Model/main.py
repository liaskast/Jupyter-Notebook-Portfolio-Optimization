# Import package of BQL to connect the database
import bql

# Create the database connection 
bq = bql.Service()

# The query pulling out the 20-day average price for SP500 members

# Define px as a price series for last 20 days
px = bq.data.px_last(dates=bq.fumc.range('-20d', '0d'))
# Define the avg_px by applying function avg to the time series px
avg_px = bq.func.avg(px)

# Define the universe as index members of SP500
index_memb = bq.univ.members('SPX Index')

# Create the request
request = bql.Request(index_memb, {'avg_px': avg_px}, with_params={'currency': 'USD'})

# Execute the rquiest with database conncection
response = bq.execute(request)

# combined_df os a utility function to convert response object into panda's DataFrame
data = bql.combined_df(response)    