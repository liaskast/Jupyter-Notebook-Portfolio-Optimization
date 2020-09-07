# Black-Litterman Model for Portfolio Weight Optimization

The Black–Litterman model is a mathematical model for portfolio allocation. The model expands upon the de facto standard for portfolio optimization, **Markowitz's classical Modern Portfolio Theory** by incorporating the opinion of the investor regarding future asset performance and modifying the allocation accordingly.

## Getting Started

### How to Run

#### Virtual Execution

Please find below, a link to a [virtual workspace](https://mybinder.org/v2/gh/liaskast/Jupyter-Notebook-Portfolio-Optimization/master) where the porfolio optimization can be launched.

[![badge](https://img.shields.io/badge/Launch-Virtual%20Workspace-F5A252.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/liaskast/Jupyter-Notebook-Portfolio-Optimization/master)

#### Local Execution

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
