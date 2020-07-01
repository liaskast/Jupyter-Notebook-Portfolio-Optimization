# Import the Bloomberg Query Language (BQL) and bqfactor libraries
#import bql
#import bqport
# Import other data analytics and chatting libraries
import pandas as pd
import bqplot as bqp
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

    class port:
        def list_portfolios():
            mytuple = ("apple", "banana", "cherry")
            return mytuple

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
    #print("\n Initial Optimal Weights")
    #return W_opt

solve_intial_opt_weight() # Here we call the Optimization function that returns the optimal weights.
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
def onclickapplysettings(obj=None):
    save_settings()
    updateinputboxes()
    solve_intial_opt_weight()
    updateviewcontrol()
    updatecontrolinui()
    run_viewmodel({'new':0.})
    
button_applysettings.on_click(onclickapplysettings)
#UI_sec_input = HBox([VBox(list_sec_input),VBox([load_members_hbox,label_usemktcap,check_usemktcap,label_usemktcap2,button_applysettings],layout={'margin':'0px 0px 0px 10px'})])
UI_sec_input = HBox([VBox(list_sec_input),VBox([label_usemktcap,check_usemktcap,label_usemktcap2,button_applysettings],layout={'margin':'0px 0px 0px 10px'})])

def on_click_load_portfolio(obj=None):
    global df_portfolio_weight
    #portfolio_univ = bq.univ.members(port_dict[portfolio_dropdown.value],type='PORT') #********************TALKS TO Bloomberg's Database************ xox
    #id_ = bq.data.id() #********************TALKS TO Bloomberg's Database************ xox
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
            alpha = (0.5 - Pi[n][0]) * (floattext_confidence.value)
            Q.append(alpha + Pi[n][0])

        '''
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
        '''
        Q=np.array([Q]).T 
        #tau = floattext_uncertainty.value 
        tau = 1/(5*12-len(list_security)) #tau is a scalar indicating the uncertainty 

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
        
        line.x = frontier['risk']
        line.y = [frontier['return'],new_frontier['return']]
        


floattext_confidence = FloatSlider(description='Confidence Level on Views', value=dict_settings['confidence'],style={'description_width':'initial'}, readout_format='.2%', max=1, min=0,
                                   layout={'margin':'20px 0px 0px 0px'}, step=0.5/100)

floattext_confidence.observe(run_viewmodel) 

#sv = pd.Series(np.sqrt(np.diag(Pi.T.dot(C.dot(Pi))).astype(float)), index=C.index)
def updateviewcontrol():
    global UI_viewcontrol, list_slider, list_relative_controls, floattext_uncertainty
    
    list_slider=[]
    list_security=list(dict_settings['security'].keys())
    for n in range(len(dict_settings['security'])):
        temp_slider=FloatSlider(value=Pi[n], description=list_security[n], max=0.2, min=-0.2, readout_format='.2%', step=0.2/100,style={'description_width':'100PX'})
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

bar = bqp.Bars(x=[], 
               y=[], 
               scales={'x': x_ord, 'y': y_sc},
               orientation="horizontal", display_legend=True, labels=['Mkt Efficient Portfolio','Efficient Portfolio with Views'],
              colors=['#1B84ED','#F39F41'],
              type='grouped')
#bar.type='grouped'
bar.tooltip = bqp.Tooltip(fields=['y'], labels=['Weights'], formats=['.3f'])

ax_x = bqp.Axis(scale=x_ord, orientation="vertical")
ax_y = bqp.Axis(scale=y_sc, label='Weight')

fig_bar = bqp.Figure(marks=[bar], axes=[ax_x, ax_y], padding_x=0.025, padding_y=0.025, 
                     layout=Layout(width='800px'), legend_location='top-right', 
                     fig_margin={'top':20, 'bottom':30, 'left':80, 'right':20})

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

#import bqcde #********************TALKS TO Bloomberg's Database************ AND BELOW...interal portfolio thing
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
button = Button(description='Upload to CDE')
button.on_click(upload_to_cde)

# END OF ************************************************************************************************************** (use of bqcde)  *********************** 


#preload_box.children = []
VBox([button,tab])

#[Open Weight](output.xlsx)

for slider in list_slider:
    print(slider.description, ": ", slider.value)        