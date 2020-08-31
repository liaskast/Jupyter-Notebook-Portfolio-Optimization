# Black-Litterman Model for Portfolio Weight Optimization

The Black–Litterman model is a mathematical model for portfolio allocation. The model expands upon the de facto standard for portfolio optimization, **Markowitz's classical Modern Portfolio Theory** by incorporating the opinion of the investor regarding future asset performance and modifying the allocation accordingly.

## Instructions

### How to Run

#### Virtually

Please find a link to a virtual workspace where the porfolio optimization can be launched.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/liaskast/Jupyter-Notebook-Portfolio-Optimization/master)

#### Locally

The main piece of code that drives the notebook can be found in file [main](Main/main.py), while the notebook where the code is run is called [Terminal](Main/Terminal.ipynb). 

### How to Input your Own Portfolio

N.B. To determine at which location in the code to find each of the below steps please find in the code (i.e. ctrl+f) the following commands.

**Step**: #1 - Input the assets to portfolio.

Description: To specify which assets are included in our portfolio we need to provide each product in the form below:
```python
security['Name'] =  'Ticker'
```
**Step**: #2 - Input the weights of the portfolio.

Description: To input the weights of each asset class please provide the weights of each product in the form below:
```python
approximated_mkt_weight = [weight_of_asset_class_no1,weight_of_asset_class_no2,weight_of_asset_class_no3,...]
```
N.B.: make sure that the same number of weights as asset classes are provided other wise you will receive Error #1.

**Step**: #3 - Read in Asset Classes from Excel.

Description: To read in asset classes Directly from Excel the user has to specify the file name from which they have to be read as well as the number of columns from that excel file that contain data. This is performed by the following line of code:
```python
prices = pd.read_excel ('filename.xlsx',header=1,index_col=0, parse_dates= True, usecols="A:N")
```
N.B. By default the excel file that contains the data series provided as input should be placed in the same directory as the Jupyter Notebook that uses it as input. We call file "filename.xlsx" but the user provide their own file name by also chaning it in the above line of code. Parameter "usecols" specifies  which columns are read-in by the program. It should be column "A" until "last_column_of_data + 1".

### How to Input a View

- **Using the Slider**: Simply drag the slider to the preferred value.
- **Type in Required Value**: Double click on the "view" attribute and then input the number required and finally press enter to confirm. N.B. the user is required to type in the value in percentage form (%), i.e. when required to type in a value of 5%, the user must type in  0.05 instead.

## Common Errors and Solutions

Below is a collection of commons errors faced during the development phase of this product and the respective solutions that frequently resolved those. 

**#1**
```python
ValueError: cannot reshape array of size 4 into shape (3,1)
```
Explanation: 
 - You have read wrong data from Excel. i.e. you have read in more/less columns than required.
 - You have included more/less weights than the number of products in your portfolio.

Solution: 
 - Check the length of the array of weights, and/or the length of the products in your portfolio.
