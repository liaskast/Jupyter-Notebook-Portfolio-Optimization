# Import the Bloomberg Query Language (BQL) and bqfactor libraries
#import bql
#import bqport
# Import other data analytics and chatting libraries


from __future__ import print_function
import pandas as pd
import bqplot as bqp
import math as math
#import bqviz as bqv
import numpy as np
from numpy.linalg import inv
from collections import OrderedDict
import scipy
import ipywidgets as widgets
from scipy import optimize

#import bqwidgets as bqw
from ipywidgets import HBox, VBox, IntSlider, Text, Tab, FloatText, Label, Layout, FloatText, IntText, Checkbox, Button, FloatSlider, Dropdown, HTML # python library that contains items such labels, checkbox
from IPython.display import display # Required so that we are able to display widgets and other code at the same time otherwise widgets are supressed and not displayed.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Loading animation
loading_html = HTML("""
    <div style="font-size:14px; color:lightskyblue;">
        <i class="fa fa-circle-o-notch fa-spin"></i><span>&nbsp;Loading...</span>
    </div>""")

preload_box = HBox([loading_html])
preload_box
#display(preload_box) # Required to force the system to display the preload_box.

from bqplot import pyplot as plt
from bqplot import topo_load
from bqplot.interacts import panzoom
# initializing data to be plotted
np.random.seed(0)
size = 100
y_data = np.cumsum(np.random.randn(size) * 100.0)
y_data_2 = np.cumsum(np.random.randn(size))
y_data_3 = np.cumsum(np.random.randn(size) * 100.)

x = np.linspace(0.0, 10.0, size)
#plt.figure(title='Scatter Plot with colors')
#plt.scatter(y_data_2, y_data_3, color=y_data, stroke='black')
#plt.show();

# Instantiate an object to interface with the BQL service
#bq = bql.Service() # object bq is defined #Requires Bloomberg's Database and is henceforth unaccessible to us.

class Bloomberg_Object:
    class data:
        def day_to_day_total_return(self):
            print("")
        def NAME(self):
            print("")    

    class execute:
        print("")

    class univ:
        print("")

    class port:
        def list_portfolios():
            mytuple = ("apple", "banana", "cherry")
            return mytuple

bq = Bloomberg_Object()        

#1 - Input the assets to portfolio.
security = OrderedDict()
#APOLIS
#security['1-5 years GILTS'] =  'SYB5 GY Equity'
#security['Cash'] =  'BNPIEGI LX Equity'
#security['Chinese Equity'] =  'SHSZ300 Index'
#security['Emerging Asia Equity'] =  'NDUEEGFA Index'
#security['EU High Yield Bonds'] =  'EUNW GY Equity'
#security['European Banks'] =  'SX7E Index'
#security['European Corp'] =  'EUN5 GR Equity'
#security['European Equity'] =  'SXXE Index'
#security['German Equity'] =  'DAX Index'
#security['Greek Equity'] =  'FTASE Index'
#security['Greek Govies'] =  'BEGCGA Index'
#security['Italian Equity'] =  'FTSEMIB Index'
#security['MSCI Info tech'] =  'NDWUIT Index'
#security['MSCI World'] =  'MACXUIGB Index'
#security['Spanish Equity'] =  'IBEX Index'
#security['US Equity'] =  'SPX Index'
#security['US High Yield Bonds'] =  'IBXXHYCT Index'

#Benchmarks
#security['Euro Gov'] =  'Euro Gov'
#security['Greek Gov'] =  'Greek Gov'
#security['EU Corporate'] =  'EU Corporate'
#security['EU High Yield'] =  'EU HY'
#security['European Equities'] =  'Eur Eq'
#security['US Equities'] =  'US Eq'
#security['Cash'] =  'Cash'

#Benchmarks 2 - TEA
#security['LEATTREU Index'] =  'EU Gov'
#security['LEC4TREU Index'] =  'EU Corps'
#security['BEGCGA Index'] =  'GR Corps'
#security['SXUSR Index	US'] =  'US Equity'
#security['SX5R Index'] =  'EU Equity'
#security['LEF1TREU Index'] =  'FRN EU'
#security['EUR001M Index'] =  'Cash'

#Benchmarks 3 - TEA ETFs
security['EUNH GY Equity'] =  'EU Gov'
security['EUN5 GY Equity'] =  'EU Corps'
security['LFGGBDR LX Equity'] =  'GR Corps'
security['SPY US Equity'] =  'US Equity'
security['SX5EEX GY Equity'] =  'EU Equity'
security['FLOT FP Equity'] =  'FRN EU'
security['PARSTEI LX Equity'] =  'Cash'

#Bloomberg
#security['S&P 500'] = 'SPY US Equity'
#security['Real Estate'] = 'IYR US Equity'
#security['Russ 1K Val'] = 'IWD US Equity'
#security['Small Stocks'] = 'IWM US Equity'
#security['Commodities'] = 'DBC US Equity'
#security['GOLD'] = 'GLD US Equity'
#security['Russ 1K Gro'] = 'IWF US Equity'
#security['Bonds - Agg'] = 'AGG US Equity'
#security["Int'l Bonds"] = 'BWX US Equity'
#security["High Yield"] = 'HYG US Equity'
#security["US Treasuries"] = 'GOVT US Equity'
#security["Emerging Mkts"] = 'EEM US Equity'

#2 - Input the weights of the portfolio.
#Original Weights that will provide us with the Implied returns. They only infulence the Implied returns' values and not the allocation policy. Even if originally we allocate 100% of our portfolio to one asset class the code will allocate according to the returns and will not be influenced by our weights, the weights only affect the implied returns.

#Experimental Weights
# Weights for a portfolio Without Gilts and Chinese Equity.
#approximated_mkt_weight = [0.00957643167488187,0.010241765265639,0.398894134073001,0.00416351972379412,0.0967099088024052,0.0828703866165383,0.0235103219298358,0.0125595027532384,0.0120035820663699,0.0106296429781949,0.0202795023703381,0.035435880040154,0.00992384006540524,0.0311647410666334,0.0659143843855553]
#approximated_mkt_weight = [0.0112878580039961,0.164879596149528,0.00957643167488187,0.010241765265639,0.398894134073001,0.00416351972379412,0.0967099088024052,0.0828703866165383,0.0235103219298358,0.0125595027532384,0.0120035820663699,0.0106296429781949,0.0202795023703381,0.035435880040154,0.00992384006540524,0.0311647410666334,0.0659143843855553]
# Weights for a portfolio without Cash
#approximated_mkt_weight = [0.0112878580039961,0.00957643167488187,0.010241765265639,0.398894134073001,0.00416351972379412,0.0967099088024052,0.0828703866165383,0.0235103219298358,0.177439098902766,0.0120035820663699,0.0106296429781949,0.0202795023703381,0.035435880040154,0.00992384006540524,0.0311647410666334,0.0659143843855553]

#print(len(approximated_mkt_weight)) # Prints the size of weights to confirm the right amount of products.

# Original Apolis Weights
#approximated_mkt_weight = [0.0112878580039961,0.164879596149528,0.0248550020344915,0.00957643167488187,0.010241765265639,0.398894134073001,0.00416351972379412]
#approximated_mkt_weight = [0.0112878580039961,0.164879596149528,0.0248550020344915,0.00957643167488187,0.010241765265639,0.398894134073001,0.380265213]
#approximated_mkt_weight = [0.1465,0.2869,0.21863,0.214563,0.114563,0.11463,0.1146]
#approximated_mkt_weight = [0.01,0.16,0.024,0.00957,0.010241,0.39889,0.380265]
# Weights for Original Portfolio
#approximated_mkt_weight = [0.14,0.02, 0.15, 0.01,0.05,0.05,0.1, 0.05, 0.20, 0.05, 0.15, 0.03]
# Example Weights for attempt with 4 products
#approximated_mkt_weight = [0.14,0.02, 0.15, 0.01]
# TEA - weights
approximated_mkt_weight = [0.3,0.2, 0.15, 0.1,0.1,0.05,0.1]
#approximated_mkt_weight = [0.35,0.2, 0.15, 0.1,0.2]

#approximated_mkt_weight = [0.3,0.2, 0.15, 0.1,0.1,0.05,0.1]
#approximated_mkt_weight = [0.3,0.3, 0.15, 0.1,0.1,0.05]


dict_settings = OrderedDict()
dict_settings['security'] = security
dict_settings['weight'] = approximated_mkt_weight

#rf = 0.015 # Original line of code - rf is the risk-free rate
rf = 0 # New line of code - rf is the risk-free rate
num_avail_ticker=len(dict_settings['security'])
#print(len(dict_settings['security'])) # prints the number of securities considered. This is used as a test to see whether the right portfolio is read as input.
uncertainty = 0.025 # tau is a scalar indicating the uncertainty in the CAPM (Capital Asset Pricing Model), this is a parameter for Black-Litterman

#******************************************************************************** Reads in Input ****************************************************************************************************
#3 - Read in Asset Classes from Excel.
prices = pd.read_excel ('prices.xlsx',header=1,index_col=0, parse_dates= True, usecols="A:H") # usecols: specifies  which columns are read-in by the program. It should be column "A" until "last_column + 1".
returns = prices.pct_change()
returns = returns.dropna()

import pickle
from collections import OrderedDict
from datetime import timedelta

#Creates pickle file called 'settings_bl.pckl'. This file 'serializes' python Objects.
#try: 
    #f = open('settings_bl.pckl', 'rb') #**************************************************Issue?
    #dict_settings = pickle.load(f)
    #f.close()  
      
#except:                         # Defines Python Objects.  
    #dict_settings = OrderedDict()
    #dict_settings['security'] = security
    #dict_settings['weight'] = approximated_mkt_weight
    #dict_settings['confidence'] = 0.8
    #dict_settings['scalar'] = uncertainty
    #dict_settings['usemktcap'] = False # Here we define the option to use the mkt cap as weighting if you choose index securities. We will not use!!!

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
    #print("entered bq_ref_data")
    def merge(response): 
        return pd.concat([sir.df()[sir.name] for sir in response], axis=1)
    result=merge(response)
    return result

def bq_series_data(security,datafields):
    #request =  bql.Request(security, datafields) #******************** Directly TALKS TO Bloomberg's Database ************
    #request = 1
    #response = bq.execute(request) #******************** Directly TALKS TO Bloomberg's Database ************
    #print("entered bq_series_data")
    response = returns
    #result = response[0].df().reset_index().pivot(index='DATE',columns='ID',values=response[0].name)[security]
    return response

# Portfolio Mean
def _port_mean(weights, expected_returns):
    if((expected_returns.T * weights).sum()>0.02): # This is where we place a bound on the return provided by the portfolio.
        return((expected_returns.T * weights).sum())
    else:
        return 1

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
    weights.index.name = 'portfolio'
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
    R = day_to_day_return.dropna().mean()*52 # Input: 252 yearly  # Description:  R is the vector of expected returns in the "Step-by-Step Guide..." paper
    C = day_to_day_return.cov()*52 # Input: 252 yearly # Description: C is the covariance matrix i.e. Sigma in the "Step-by-Step Guide..." paper
    
    if dict_settings['usemktcap']: # This is the option to use the mkt cap as weighting if you choose index securities. We will not use!!!
        datafields = OrderedDict()
        datafields['Mkt Cap'] = bq.data.cur_mkt_cap(currency='usd') # Datafields Parameter
        df_mkt_cap=bq_ref_data(univ,datafields)  # Gets mkt_cap of the security in the Tickers. 
        W = np.array(df_mkt_cap/df_mkt_cap.sum()) # W is the market cap weight
    else:
        W = np.array(dict_settings['weight']).reshape(len(R),1)
        
    new_mean = _port_mean(W.T[0],R) # this variable is not used
    new_var = _port_var(W,C) # this variable is not used

    lmb = 0.5/np.sqrt(W.T.dot(C).dot(W))[0][0] # Compute implied risk adversion coefficient: 1/(2*(wΣw))
    Pi = np.dot(lmb * C, W) # Compute equilibrium excess returns

    frontier, f_weights = solve_for_frountier(Pi+rf, C, rf)
    frontier['sharpe']=frontier['return']/frontier['risk']
    f_weights.columns = R.keys()
    # Solve for weights before incorporating views
    W_opt = np.array(f_weights.iloc[frontier.loc[frontier['sharpe']==frontier['sharpe'].max()].index.values[0]])
    mean_opt, var_opt = _port_mean_var(W_opt, Pi+rf, C)   # calculate tangency portfolio
    print(mean_opt*100)
    print(math.sqrt(var_opt)*100)
    #print("\n Initial Optimal Weights")
    #print(W_opt)
    return W_opt, frontier, f_weights, Pi, C


solve_intial_opt_weight() # Here we call the Optimization function that returns the initial optimal weights.
# ************************************************************************************* GUI portion of the code that contains various labels,checkboxes etc.  ***********************

input_header = HBox([Label(value='Ticker', layout=Layout(width='120px',height='22px')), Label(value='Name of Asset', layout=Layout(width='120px',height='22px')), 
                     Label(value='Weight',layout=Layout(width='120px',height='22px'))])
list_sec_input = [input_header]
lst_name = list(dict_settings['security'].keys())
lst_ticker = list(dict_settings['security'].values())
lst_weight = dict_settings['weight']

# Checkbox that determines the mktcap option. Places in python object's 'check_usemktcap' value field a true or false value
check_usemktcap = Checkbox(description='Use Market Cap as Weight',value=dict_settings['usemktcap'], layout=Layout(min_width='15px'),style={'description_width':'initial'}) 

label_usemktcap = Label(value=' ',layout={'height':'22px'}) # MktCap Label
label_usemktcap2 = Label(value='(not recommended when using ETF as proxy)',layout=Layout(min_width='300px')) # MktCap Label

# Option to load your own internal portfolio different from the original input.

#port_dict = {x['name']: x['id'].split(':')[2].replace('-','U') + '-' + x['id'].split(':')[3] for x in bqport.list_portfolios()}
#port_dict = {x['name']: x['id'].split(':')[2].replace('-','U') + '-' + x['id'].split(':')[3] for x in bq.port.list_portfolios()}
#load_button = Button(description='Load members')
#portfolio_dropdown = Dropdown(description='Portfolio:',options=sorted(set(port_dict.keys())))
#load_members_hbox = HBox([portfolio_dropdown, load_button])


for n in range(num_avail_ticker):
    text_name = Text(layout=Layout(width='120px'))
    text_ticker = Text(layout=Layout(width='120px'))

    list_sec_input.append(HBox([text_ticker, text_name, FloatText(layout=Layout(width='120px'))]))
    try:
        if check_usemktcap.value:
            list_sec_input[n+1].children[2].disabled = True
            list_sec_input[n+1].children[2].layout.visibility = 'hidden'
        list_sec_input[n+1].children[0].value = lst_ticker[n]
        list_sec_input[n+1].children[1].value = lst_name[n] if lst_name[n] != '' else list_sec_input[n+1].children[0].data[0].split(":")[-1].strip() 
        list_sec_input[n+1].children[2].value = lst_weight[n]

    except:
        pass
        
def updateinputboxes(obj=None):
    lst_name = list(dict_settings['security'].keys())
    lst_ticker = list(dict_settings['security'].values())
    for n in range(num_avail_ticker): 
        if list_sec_input[n+1].children[0].value.strip() != '' and list_sec_input[n+1].children[1].value.strip() == '':
            list_sec_input[n+1].children[1].value = lst_name[n]
        if check_usemktcap.value:
            list_sec_input[n+1].children[2].disabled = True
            list_sec_input[n+1].children[2].layout.visibility = 'hidden'
        else:
            list_sec_input[n+1].children[2].disabled = False
            list_sec_input[n+1].children[2].layout.visibility = 'visible'

check_usemktcap.observe(updateinputboxes, 'value')

button_applysettings=Button(description = 'Apply Settings')
button_reset=Button(description = 'Reset to No Views')
def onclickapplysettings(obj=None):
    save_settings()
    updateinputboxes()
    solve_intial_opt_weight()
    updateviewcontrol()
    updatecontrolinui()
    run_viewmodel({'new':0.})
    
display(button_reset)
button_reset.on_click(onclickapplysettings)
button_applysettings.on_click(onclickapplysettings)

#UI_sec_input = HBox([VBox(list_sec_input),VBox([load_members_hbox,label_usemktcap,check_usemktcap,label_usemktcap2,button_applysettings],layout={'margin':'0px 0px 0px 10px'})])
UI_sec_input = HBox([VBox(list_sec_input),VBox([label_usemktcap,check_usemktcap,label_usemktcap2,button_applysettings],layout={'margin':'0px 0px 0px 10px'})]) # Have taken load_members_hbox out because it requires a call to bloomberg's cde library which is not accessible to us.

def on_click_load_portfolio(obj=None):
    global df_portfolio_weight
    #portfolio_univ = bq.univ.members(port_dict[portfolio_dropdown.value],type='PORT') #Requires Bloomberg's Database and is henceforth unaccessible to us.
    #id_ = bq.data.id() #Requires Bloomberg's Database and is henceforth unaccessible to us.
    #df_portfolio_weight = pd.concat([x.df() for x in bq.execute(bql.Request(portfolio_univ, [bq.data.name(),id_['Weights']/100]))],axis=1).reset_index()  #******************** Directly TALKS TO Bloomberg's Database ************
    df_portfolio_weight =  approximated_mkt_weight
    for x in range(1,num_avail_ticker+1):
        if x - 1 < len(df_portfolio_weight):
            list_sec_input[x].children[0].value = df_portfolio_weight.iloc[x-1,0]
            list_sec_input[x].children[1].value = str(df_portfolio_weight.iloc[x-1,1])
            list_sec_input[x].children[2].value = df_portfolio_weight.iloc[x-1,2]
        else:
            list_sec_input[x].children[0].value = ''
            list_sec_input[x].children[1].value = ''
            list_sec_input[x].children[2].value = 0

#on_click_load_portfolio()
#load_button.on_click(on_click_load_portfolio)

def run_viewmodel(change=None):
    # VIEWS ON ASSET PERFORMANCE
    
    # for troubleshoot
    global sub_a, sub_b, sub_c, sub_d, tau, omega, P, Q, Pi_new
    list_security=list(dict_settings['security'].keys())
    weights=OrderedDict()
    
    P=np.identity(len(dict_settings['security']))

    if isinstance(change['new'],float):
        Q=[]
        for n in range(len(dict_settings['security'])):
            alpha = (list_slider[n].value - Pi[n][0]) * (floattext_confidence.value) # We supply the [primary 'view'] which is the Implied Return for each Asset Class.
                                                                                     # For example, we assume the first asset class will have implied return 6% as calculated from the historical data
                                                                                     # then, list_slider[n].value contains that 6% value. This is the [primary 'view'] for this asset class. 
                                                                                     # Then on top of this [primary 'view'] we will add our own [portfolio manager 'view']. 
                                                                                     # For example we believe that this asset class instead of 6% will have a 8% return. 
                                                                                     # Hence, we move the 'slider' from the 6% position to the 8% position. This is done later inside function "updateviewcontrol"
            Q.append(alpha + Pi[n][0])

        
        for relative_box in list_relative_controls:
            sec1_pos = relative_box.children[0].value - 1
            sec2_pos = relative_box.children[2].value - 1
            if sec1_pos >= 0 and sec2_pos >= 0:
                npselection = np.zeros(len(dict_settings['security']))
                npselection[sec1_pos] = 1
                npselection[sec2_pos] = -1
                P = np.array(pd.DataFrame(P).append(pd.DataFrame(npselection).T))
                alpha = (relative_box.children[-1].value - (Pi[sec1_pos][0] - Pi[sec2_pos][0])) * (floattext_confidence.value)
                Q.append(alpha +  (Pi[sec1_pos][0] - Pi[sec2_pos][0]))
        
        Q=np.array([Q]).T 
        #tau = floattext_uncertainty.value 
        tau = 1/(5*12-len(list_security)) #tau is a scalar indicating the uncertainty. As specified in the Black-Litterman paper it should be anywhere between [0.01,0.05] and this formula guarantees that in the majority of cases. 
        #tau = 0.025 # this allows us to fix the uncertainty value to a pre-determined standard value, equal to the above formula when 20 asset classes are supplied as input..
        omega = np.dot(np.dot(np.dot(tau, P), C), P.T)# omega represents uncertanity of views implied uncertainty from market parameters.

        # Compute equilibrium excess returns taking into account views on assets
        sub_a = inv(np.dot(tau, C))
        sub_b = np.dot(np.dot(P.T, inv(omega)), P)
        sub_c = np.dot(inv(np.dot(tau, C)), Pi)
        sub_d = np.dot(np.dot(P.T, inv(omega)), Q)
        Pi_new = np.dot(inv(sub_a + sub_b), (sub_c + sub_d))         
        # Perform a mean-variance optimization taking into account views   
        new_frontier, new_f_weights = solve_for_frountier(Pi_new + rf, C, rf)
        new_frontier['sharpe']=new_frontier['return']/new_frontier['risk']
        # Solve for weights before incorporating views
        new_weights = np.array(new_f_weights.iloc[new_frontier.loc[new_frontier['sharpe']==new_frontier['sharpe'].max()].index.values[0]])

        leverage = np.sum(abs(new_weights))
        weights['Initial Weights']=approximated_mkt_weight[::-1]
        weights['Opt Portfolio']=W_opt[::-1]
        weights['Opt Portfolio with View']=new_weights[::-1]
        
        output_df = pd.DataFrame(weights, index=list_security[::-1])
        output_df.to_excel('output.xlsx')

        mean, var = _port_mean_var(new_weights[::-1], Pi_new + rf, C)
        scatt.x = [np.sqrt(var_opt)]
        scatt.y = [mean_opt]
        scatt_view.x = [np.sqrt(var)]
        scatt_view.y = [mean]
            
        bar.x = list_security[::-1]
        bar.y = [weights[col] for col in weights]

        labels.y = list_security[::-1]
        labels.x = weights['Initial Weights']*100
        labels.text = weights['Initial Weights']

        #bar_labels.x = list_security[::-1]
        #bar_labels.y = [weights[col] for col in weights]
        
        line.x = frontier['risk']
        line.y = [frontier['return'],new_frontier['return']]
        


floattext_confidence = FloatSlider(description='Confidence Level on Views', value=dict_settings['confidence'],style={'description_width':'initial'}, readout_format='.2%', max=1, min=0,
                                   layout={'margin':'20px 0px 0px 0px'}, step=0.5/100)
                                   
floattext_confidence.observe(run_viewmodel) 

#sv = pd.Series(np.sqrt(np.diag(Pi.T.dot(C.dot(Pi))).astype(float)), index=C.index)
def updateviewcontrol():
    global UI_viewcontrol, list_slider, list_relative_controls, floattext_uncertainty
    
    list_slider=[]
    #list_security=list(dict_settings['security'].keys()) # Original line
    list_security=list(dict_settings['security'].values()) # Changed the name next to sliders to ticker name.
    for n in range(len(dict_settings['security'])):
        #temp_slider=FloatSlider(value=Pi[n], description=list_security[n], continuous_update=False, max=0.2, min=-0.2, readout_format='.2%', step=0.2/100,style={'description_width':'100PX'})
        temp_slider=FloatSlider(value=Pi[n], description=list_security[n], max=0.2, min=-0.2, readout_format='.2%', step=0.2/100,style={'description_width':'100PX'}) #Slider Specficiations. Pi[n] contains the [primary 'view'] and is the starting point of the slider. max,min specify the maximum amount of return you can spec on an asset class. description=list_security[n]--> contains the name attached to the left of each slider.
        #display(temp_slider) # this command was required to forcefully display sliders when bqplot did not use to work. It is no longer required as Bqplot now works on the jupyter notebook.
        temp_slider.observe(run_viewmodel)
        list_slider.append(temp_slider)
    
    list_relative_controls=[]
    sec_dropdown_options = OrderedDict(zip(['None']+list(dict_settings['security'].keys()),range(len(dict_settings['security'])+1)))
    
    for n in range(3):
        dropdown1 = Dropdown(options=sec_dropdown_options,layout={'width':'100px'})
        dropdown2 = Dropdown(options=sec_dropdown_options,layout={'width':'100px'})
        label = Label(value='over',layout={'width':'30px'})
        float_value = FloatSlider(description='by', value=0, readout_format='.2%', max=0.2, min=0,
                                           style={'description_width':'initial'}, step=0.1/100,layout={'width':'200px'})
        float_value.observe(run_viewmodel)
        relative_box = HBox([dropdown1,label,dropdown2,float_value])
        list_relative_controls.append(relative_box)
    
    header_abs_html = HTML('<p style="color: white;">{}</p>'.format('Absolute Views'))
    header_rel_html = HTML('<p style="color: white;">{}</p>'.format('Relative Views'), layout={'margin':'20px 0px 0px 0px'})
    UI_viewcontrol = [header_abs_html, VBox(list_slider),header_rel_html, VBox(list_relative_controls), VBox([floattext_confidence])]
    
    
def updatecontrolinui():
    UI_model.children[0].children = UI_viewcontrol

updateviewcontrol()

# END OF ************************************************************************************* GUI portion of the code that contains various labels,checkboxes etc.  ***********************
# START OF ************************************************************************************************************** Build bar charts (use of bqp)  ***********************

x_ord = bqp.OrdinalScale()
y_sc = bqp.LinearScale()

#Plot #1 i.e. Creation of the bar plot

bar = bqp.Bars(x=[], 
               y=[], 
               scales={'x': x_ord, 'y': y_sc},
               orientation="horizontal", display_legend=True, labels=['Initial Weights','Mkt Efficient Portfolio','Efficient Portfolio with Views'], #orientation decides whether the bars are horizontal or vertical
              colors=['#1B84ED','#4fa110','#F39F41'],
              type='grouped')
bar_labels = bqp.Label(x=[], y=[], scales={'x': x_ord, 'y': y_sc},    x_offset = 2, y_offset = 7, 
    text=[0,0,0], colors=['blue','blue', 'blue'], 
    default_size=24,  update_on_move=True)
#bar.type='grouped'
bar.tooltip = bqp.Tooltip(fields=['y'], labels=['Weight of Asset'], formats=['.3f']) #this displays the weight placed on each asset.

ax_x = bqp.Axis(scale=x_ord, orientation="vertical")
ax_y = bqp.Axis(scale=y_sc, label='Weight')

#fig_bar = bqp.Figure(marks=[bar], axes=[ax_x, ax_y], padding_x=0.025, padding_y=0.025, 
                     #layout=Layout(width='800px'), legend_location='top-right', 
                     #fig_margin={'top':20, 'bottom':30, 'left':80, 'right':20})

#fig_bar = bqp.Figure(marks=[bar,bar_labels], axes=[ax_x, ax_y], padding_x=0.025, padding_y=0.025, 
#                     layout=Layout(width='600px'), legend_location='top', 
#                     fig_margin={'top':20, 'bottom':30, 'left':110, 'right':20})                     

x_labels = ['aaa','bbb','ccc']
x_ord = bqp.OrdinalScale()
y_sc = bqp.LinearScale()
y_sc.max = 10
#bar_git = bqp.Bars(x= x_labels, y=[2,10,15], scales={'x': x_ord, 'y': y_sc},orientation="horizontal" )

ax_x = bqp.Axis(scale=x_ord, orientation="vertical", color = 'Black')
ax_y = bqp.Axis(scale=y_sc, tick_format='0.2f', color = 'White')

#labels_original =  bqp.Label(y=x_labels, x=[2,30,5], scales={'y': x_ord, 'x': y_sc}, 
#    x_offset = 2, y_offset = 7, 
#    text=[333,66666666,99999999], colors=['blue','blue', 'blue'], 
#    default_size=24,  update_on_move=True)
labels =  bqp.Label(y=[], x=[], scales={'y': x_ord, 'x': y_sc}, 
    x_offset = 2, y_offset = 7, 
    text=[], colors=['blue','blue', 'blue'], 
    default_size=24,  update_on_move=True)
fig_bar = bqp.Figure(marks=[labels,bar], axes=[ax_x, ax_y], padding_x=0.025, padding_y=0.025, 
                     layout=Layout(width='600px'), legend_location='top', 
                     fig_margin={'top':20, 'bottom':30, 'left':110, 'right':20})       



#Plot #2 i.e. the efficient froniter plot

x_lin = bqp.LinearScale()
y_lin = bqp.LinearScale()

x_ax = bqp.Axis(label='risk', scale=x_lin, grid_lines='solid')
x_ay = bqp.Axis(label='return', scale=y_lin, orientation='vertical', grid_lines='solid')

def_tt = bqp.Tooltip(fields=['x', 'y'], formats=['.3f', '.3f']) 

scatt = bqp.Scatter(x=[],y=[], scales={'x': x_lin, 'y': y_lin}, tooltip=def_tt,
                    display_legend=True, labels=['Efficient Portfolio'], colors=['#1B84ED'])
scatt_view = bqp.Scatter(x=[],y=[], scales={'x': x_lin, 'y': y_lin}, tooltip=def_tt,
                           display_legend=True, labels=['Efficient Portfolio with Views Portfolio'], colors=['#F39F41'])

line = bqp.Lines(x=[], y=[], scales={'x': x_lin, 'y': y_lin}, display_legend=True, labels=['Mkt Efficient Portfolio','Efficient Portfolio with Views'], colors=['#1B84ED','#F39F41'])

fig_line = bqp.Figure(marks=[line], axes=[x_ax, x_ay], 
                      legend_location='top-left', layout=Layout(width='800px'), 
                      fig_margin={'top':20, 'bottom':30, 'left':80, 'right':20})
run_viewmodel({'new':0.})
UI_model=HBox([VBox(UI_viewcontrol,layout=Layout(width='450px')),VBox([fig_bar,fig_line])])
#UI_model=HBox([VBox(UI_viewcontrol,layout=Layout(width='450px'))])
#UI_model = HBox([loading_html])


# END OF ************************************************************************************************************** Build bar charts (use of bqp)  ***********************

tab = Tab()
tab.children = [UI_model, UI_sec_input]
tab.set_title(0, 'B-L Model')
tab.set_title(1, 'Settings')
#tab.set_title(2, 'Reference Data')

def updatedseclist(obj=None):
    if obj['old'] == 1:
        save_settings()
        solve_intial_opt_weight()
        run_viewmodel({'new':0.})

# START OF ************************************************************************************************************** (use of bqcde)  *********************** 

#import bqcde #Requires Bloomberg's Database and is henceforth unaccessible to us. AND BELOW...interal portfolio thing
from datetime import date
def upload_to_cde(obj):
    obj.description = 'Uploading...'
    #lmb_2 = 0.5/np.sqrt(np.asarray([W_opt]).dot(C).dot(np.asarray([W_opt]).T))[0][0]
    #Pi_2 = np.dot(lmb_2 * C, np.asarray([W_opt]).T)
    df = pd.DataFrame({'ID':list(dict_settings['security'].values()),'Equil Return': [x[0] for x in Pi], 'BL Return':[x[0] for x in Pi_new] })
    as_of_date = date.today()
    upload_dict = {'Equil Return':'UD_EQUIL_RETURN','BL Return':'UD_BL_RETURN'}
    #pd.DataFrame({'ID':list(dict_settings['security'].keys()),'Equil Return': [x[0] for x in Pi]}).set_index('ID')
    try:
        df = df.rename(columns=upload_dict)
        list_to_upload = list(upload_dict.values())
        #fs = bqcde.get_fields(mnemonics=list_to_upload)
        df['AS_OF_DATE'] = as_of_date
        df['AS_OF_DATE'] = pd.to_datetime(df['AS_OF_DATE'])
        #bqcde.write(fields=fs, dataframe=df.fillna("N/A"))
    except Exception as e:
        print(e)
    obj.description = 'Upload to CDE'

#upload_to_cde()
#button = Button(description='Upload to CDE')
#button.on_click(upload_to_cde)

# END OF ************************************************************************************************************** (use of bqcde)  *********************** 

preload_box.children = []
#VBox([button,tab])
VBox([tab])

#[Open Weight](output.xlsx)

for slider in list_slider:
    print(slider.description, ": ", slider.value)        
