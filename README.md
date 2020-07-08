# Black Litterman Model for Portfolio Weight Optimization

## Instructions

### How to Input your Own Portfolio

N.B. To locate at which location in the code to find each of the below steps please find in the code (i.e. ctrl+f) the following commands.

**command**: #1 - Input the assets to portfolio.

Description: To specify which assets are included in our portfolio we need to provide each product in the form below:

security['Name'] =  'Ticker'

**command**: #2 - Input the weights of the portfolio.

Description: To input the weights of each asset class please provide the weights of each product in the form below:

approximated_mkt_weight = [weight_of_asset_class_#1,weight_of_asset_class_#2,weight_of_asset_class_#3,...]

N.B.: make sure that the same number of weights as asset classes are provided other wise you will receive Error #1.

### Common Errors 

**#1**
Error Print Out: ValueError: cannot reshape array of size 4 into shape (3,1)

Explanation: 
 - You have read wrong data from Excel. i.e. you have read in more/less columns than required.
 - You have included more/less weights than the number of products in your portfolio.

Solution: 
 - Check the length of the array of weights, and/or the length of the products in your portfolio.
