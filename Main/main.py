# Import the Bloomberg Query Language (BQL) and bqfactor libraries
#import bql
#import bqport
# Import other data analytics and chatting libraries
import pandas as pd
#import bqplot as bqp
#import bqviz as bqv
import numpy as np
from numpy.linalg import inv
from collections import OrderedDict
import scipy
from scipy import optimize

#import bqwidgets as bqw
from ipywidgets import HBox, VBox, IntSlider, Text, Tab, FloatText, Label, Layout, FloatText, IntText, Checkbox, Button, FloatSlider, Dropdown, HTML # python library that contains items such labels, checkbox
#import bqplot as bqp

# Loading animation
#loading_html = HTML("""
#    <div style="font-size:14px; color:lightskyblue;">
#        <i class="fa fa-circle-o-notch fa-spin"></i><span>&nbsp;Loading...</span>
#    </div>""")

#preload_box = HBox([loading_html])
#preload_box

# Instantiate an object to interface with the BQL service
#bq = bql.Service() # object bq is defined #********************TALKS TO Bloomberg's Database************

class Bloomberg_Object:
    class data:
        def day_to_day_total_return(self):
            print("Runs 1.")
        def NAME(self):
            print("Runs 1.")    

    class execute:
        print("Runs 2.")

    class univ:
        print("Runs 3.")

bq = Bloomberg_Object()        

#Default settings
security = OrderedDict()
'''
security['1-5 years GILTS'] =  'LF56TRGU Index'
security['Cash'] =  'DBDCONIA Index'
security['Chinese Bonds'] =  'I32561US Index'
security['Chinese Equity'] =  'SHSZ300 Index'
security['Emerging Asia Equity'] =  'NDUEEGFA Index'
security['EU High Yield Bonds'] =  'EUNW GY Equity'
security['European Banks'] =  'SX7E Index'
security['European Corp'] =  'EUN5 GR Equity'
security['European Equity'] =  'SXXE Index'
security['German Equity'] =  'DAX Index'
security['Greek Equity'] =  'FTASE Index'
security['Greek Govies'] =  'BEGCGA Index'
security['Italian Equity'] =  'FTSEMIB Index'
security['MSCI Info tech'] =  'NDWUIT Index'
security['MSCI World'] =  'MACXUIGB Index'
security['Spanish Equity'] =  'IBEX Index'
security['US Equity'] =  'SPX Index'
security['US High Yield Bonds'] =  'IBXXHYCT Index'

#security["Crypto Currency"] = 'GBTC US Equity'
'''

security['1-5 years GILTS'] =  'Euro Gov'
security['Cash'] =  'Greek Gov'
security['Chinese Bonds'] =  'EU Corporate'
security['Chinese Equity'] =  'EU HY'
security['Emerging Asia Equity'] =  'Eur Eq'
security['EU High Yield Bonds'] =  'US Eq'
security['European Banks'] =  'Cash'

#Original Weights that will provide us with the Implied returns.
#approximated_mkt_weight = [0.0112878580039961,0.164879596149528,0.0248550020344915,0.00957643167488187,0.010241765265639,0.398894134073001,0.00416351972379412,0.0967099088024052,0.0828703866165383,0.0235103219298358,0.0125595027532384,0.0120035820663699,0.0106296429781949,0.0202795023703381,0.035435880040154,0.00992384006540524,0.0311647410666334,0.0410143843855553]
approximated_mkt_weight = [0.0112878580039961,0.164879596149528,0.0248550020344915,0.00957643167488187,0.010241765265639,0.398894134073001,0.00416351972379412]

rf = 0.015 # rf is the risk-free rate
num_avail_ticker=7
uncertainty = 0.025 # tau is a scalar indicating the uncertainty in the CAPM (Capital Asset Pricing Model), this is a parameter for Black-Litterman

#******************************************************************************** Reads in Input ****************************************************************************************************
prices = pd.read_excel (r'C:\Users\user2\Documents\Python_files\Black_Litterman\Iolcus-Investments\Main\prices.xlsx',header=1,index_col=0, parse_dates= True, usecols="A:H")
returns = prices.pct_change()
returns = returns.dropna()

import pickle
from collections import OrderedDict
from datetime import timedelta

#Creates pickle file called 'settings_bl.pckl'. This file 'serializes' python Objects.
try: 
    f = open('settings_bl.pckl', 'rb') #**************************************************Issue?
    dict_settings = pickle.load(f)
    f.close()  
      
except:                         # Defines Python Objects.  
    dict_settings = OrderedDict()
    dict_settings['security'] = security
    dict_settings['weight'] = approximated_mkt_weight
    dict_settings['confidence'] = 0.8
    dict_settings['scalar'] = uncertainty
    dict_settings['usemktcap'] = False # Here we define the option to use the mkt cap as weighting if you choose index securities. We will not use!!!
    
def save_settings(caller=None): # Reads from Button any changes to objects.
    temp_sec, temp_weight = loadtickerfrominput()
    dict_settings['security'] = temp_sec
    dict_settings['weight'] = temp_weight
    dict_settings['confidence'] = floattext_confidence.value
    dict_settings['usemktcap'] = check_usemktcap.value # If check_usemktcap.value == true then you are using the option to use the mkt cap as weighting if you choose index securities. We will not use!!!
    f=open('settings_bl.pckl','wb')
    pickle.dump(dict_settings, f)
    f.close()
    
def loadtickerfrominput(): # Reads from Button any changes to objects.
    temp_ticker = []
    temp_name = []
    temp_weight = []
    dict_missnametickers = OrderedDict()
    flag_missingname = False
    for n in range(num_avail_ticker): 
        if bool(list_sec_input[n+1].children[0].value):
            temp_ticker.append(list_sec_input[n+1].children[0].value)
            temp_name.append(list_sec_input[n+1].children[1].value)
            if list_sec_input[n+1].children[1].value.strip() == '':
                dict_missnametickers[list_sec_input[n+1].children[0].value] = n
                flag_missingname = True
            temp_weight.append(list_sec_input[n+1].children[2].value)
    if flag_missingname:
        df_name=bq_ref_data(dict_missnametickers.keys(),{'name':bq.data.NAME()}) # Gets 'name' of the security in the Tickers supplied as input. #******************** Calls function that calls Bloomberg's Database ************
        for index,row in df_name.iterrows():
            temp_name[dict_missnametickers[index]] = row['name']
    temp_sec=OrderedDict(zip(temp_name,temp_ticker))
    
    return temp_sec, temp_weight

def bq_ref_data(security,datafields):
    # Generate the request using the sercurity variable and data item...i.e. the Tickers of financial instruments
    #request =  bql.Request(security, datafields) #******************** Directly TALKS TO Bloomberg's Database ************
    #request = 1
    #response = bq.execute(request) #******************** Directly TALKS TO Bloomberg's Database ************
    #response = np.zeros(2,3)
    print("entered bq_ref_data")
    def merge(response): 
        return pd.concat([sir.df()[sir.name] for sir in response], axis=1)
    result=merge(response)
    print(result)
    return result

def bq_series_data(security,datafields):
    #request =  bql.Request(security, datafields) #******************** Directly TALKS TO Bloomberg's Database ************
    #request = 1
    #response = bq.execute(request) #******************** Directly TALKS TO Bloomberg's Database ************
    print("entered bq_series_data")
    response = returns
    #result = response[0].df().reset_index().pivot(index='DATE',columns='ID',values=response[0].name)[security]
    #print(result)
    return response

# Portfolio Mean
def _port_mean(weights, expected_returns):
    return((expected_returns.T * weights).sum())

# Portfolio Var
def _port_var(weights, risk_matrix):
    return np.dot(np.dot(weights.T, risk_matrix), weights)

# Portfolio Mean Var
def _port_mean_var(weights, expected_returns, risk_matrix):
    return _port_mean(weights, expected_returns), _port_var(weights, risk_matrix)

# Calculates Mean Var
def _find_mean_variance(weights, expected_returns, covar, return_target):
    mean, var = _port_mean_var(weights, expected_returns, covar)
    penalty = 10 * abs(mean-return_target)
    return var + penalty

# Solve for optimal portfolio weights
def solve_weights(R, C, rf, target_r = None):
    n = len(R)
    W = np.ones([n])/n # Start optimization with equal weights
    b_ = [(0,1) for i in range(n)] # Bounds for decision variables
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. }) # Constraints - weights must sum to 1
    r = np.sum(R*W) if target_r is None else target_r
    # 'target' return is the expected return on the market portfolio
    optimized = scipy.optimize.minimize(_find_mean_variance, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x 

# Constructs Efficient Frontier
def solve_for_frountier(R, C, rf):
    frontier_mean, frontier_var , weights = [], [], []
    for r in np.linspace(R.min(), R.max(), num=15):
        weight = solve_weights(R, C, rf, r)
        frontier_mean.append(r)
        frontier_var.append(_port_var(weight, C))
        weights.append(weight)
    weights = pd.DataFrame(weights)
    weights.index.name = 'portolio'
    frontier = pd.DataFrame([np.array(frontier_mean), np.sqrt(frontier_var)], index=['return', 'risk']).T
    frontier.index.name = 'portfolio'
    return frontier, weights

# Initial Optimization of Weights...Implied Returns
def solve_intial_opt_weight():
    global W_opt, frontier, f_weights, Pi, C, lmb, new_mean, W, R, mean_opt, var_opt
    security = dict_settings['security']
    univ = list(security.values())
    datafields = OrderedDict()
    #datafields['return'] = bq.data.day_to_day_total_return(start='-5y',per='m') # Datafields Parameter
    day_to_day_return=bq_series_data(univ,datafields) #******************** Calls function that calls Bloomberg's Database ************

    R = day_to_day_return.dropna().mean()*12 #252  # R is the vector of expected returns
    C = day_to_day_return.cov() *12 #252 # C is the covariance matrix
    
    if dict_settings['usemktcap']: # This is the option to use the mkt cap as weighting if you choose index securities. We will not use!!!
        datafields = OrderedDict()
        datafields['Mkt Cap'] = bq.data.cur_mkt_cap(currency='usd') # Datafields Parameter
        df_mkt_cap=bq_ref_data(univ,datafields)  # Gets mkt_cap of the security in the Tickers. 
        W = np.array(df_mkt_cap/df_mkt_cap.sum()) # W is the market cap weight
    else:
        W = np.array(dict_settings['weight']).reshape(len(R),1)
        
    new_mean = _port_mean(W.T[0],R)
    new_var = _port_var(W,C)

    lmb = 0.5/np.sqrt(W.T.dot(C).dot(W))[0][0] # Compute implied risk adversion coefficient
    Pi = np.dot(lmb * C, W) # Compute equilibrium excess returns

    frontier, f_weights = solve_for_frountier(Pi+rf, C, rf)
    frontier['sharpe']=frontier['return']/frontier['risk']
    f_weights.columns = R.keys()
    # Solve for weights before incorporating views
    W_opt = np.array(f_weights.iloc[frontier.loc[frontier['sharpe']==frontier['sharpe'].max()].index.values[0]])
    mean_opt, var_opt = _port_mean_var(W_opt, Pi+rf, C)   # calculate tangency portfolio
    return W_opt, frontier, f_weights, Pi, C

solve_intial_opt_weight() # Here we call the Optimization function that returns the optimal weights.

