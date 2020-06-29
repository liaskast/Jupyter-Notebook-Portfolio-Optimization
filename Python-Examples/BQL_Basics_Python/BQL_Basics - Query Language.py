# The String Interface
#===========================================
# Define the request string
request = """
get(PX_HIGH - PX_LOW)
for(['AAPL US Equity'])
"""

# Execute the request
response = bq.execute(request)
#===========================================


#The Object Model
#===========================================
# Define a variable for the security 
security = "AAPL US Equity"

# Define data items for pricing fields
high = bq.data.px_high()
low = bq.data.px_low()

# Generate the request using the security variable and data items
request = bql.Request(security, high - low) # i.e. request function works with params (target, operation on target)

# Execute the request
response = bq.execute(request)
