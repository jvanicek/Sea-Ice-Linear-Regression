'''
Author: Joseph Vanicek
Date: 11/2/2017

Description:
Using March and September Artic Sea Ice data. This will use a simple linear regression to estimate when the artic sea ice will melt away completely
'''

import pandas as pd
import numpy as np, math
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt




def get_Mar_Sept_frame():
    '''
    Reads in a dataframe and create new columns for monlthly means of the March and September data, including the anomalies for those months. 

    Parameters: none
    Returns: A sliced dataframe of means and anamolies for march and september.

    '''
    df = pd.read_csv('data_79_16.csv', index_col = 0)
    df['March_means'] = df.loc[:,'0301':'0331'].mean(axis = 1)
    df['March_anomalies'] = df['March_means'] - df['March_means'].mean()
    df['September_means'] = df.loc[:,'0901':'0930'].mean(axis = 1)
    df['September_anomalies'] = df['September_means'] - df['September_means'].mean()
    return df.loc[:,'March_means':'September_anomalies']
def get_ols_parameters(series):
    '''
        Gets the parameters need for to make a prediction.
        ParametersL:
            series: a series of data
        Returns the Slope, y-intercept, R^2, and the P-value of the given sereis
    '''
    years_array = sm.add_constant(series.index.values)
    mean = series.values
    years = series.index.values
    formula = 'mean ~ years' # R-style formulas
    model = smf.ols(formula, data=series)
    results = model.fit()
    return [results.params[1], results.params[0], results.rsquared,results.f_pvalue]
def make_prediction(params, description='x-intercept:', x_name='x', y_name='y', ceiling=False):
    '''
    Makes a prediction of when the ice will melt completely based on the parameters from the ols calculation. Then prints a report of the prediction.
    Parameters:
        params: list of parameters used for prediction
        description: A description for what is being predicted
        x_name: Name for the  values on the x-axis.
        y_name: Name for the values on the y-axis
        ceiling: Determines if the y-intercept value from params will be round up to the nearest whole number
    Returns: none
    '''
    xint = -(params[1]/params[0])
    if ceiling:
        xint = math.ceil(xint)
    rsquare = str((round(params[2]*100)))+'%'
    pval = round(params[3]*100,1)
    if params[3] > 0.05:
        pval_text = 'This result is not statistically significant.'
    else:
        pval_text = 'This result is statistically significant.'
    
    
    print(description+' '+str(xint)+'\n'+
        str(rsquare)+' of variation in '+y_name+' accounted for by '+x_name+' (linear model)'+'\n'+
        'Significance level of results: '+str(pval)+'%'+'\n'+
        pval_text)
    


def make_fig_1(get_Mar_Sept_frame):
    '''
    Creates a line graph for the March and september means data and fits a regression line to the data

    Parameters:
        get_Mar_Sept_frame: dataframe with the mean and anomaly data

    Returns: none
    '''
    march = pd.Series(get_Mar_Sept_frame['March_means'])
    september = pd.Series(get_Mar_Sept_frame['September_means'])
    march_params = get_ols_parameters(march)
    september_params = get_ols_parameters(september)
    
    #March
    ax = march.plot(linestyle='-')
    ax.set_ylabel(r"NH Sea Ice Extent ($10^6$ km$^2$)")
    ax.yaxis.label.set_fontsize(24)
    xs = np.arange(1979, 2016)
    ys = march_params[0] * xs + march_params[1] 
    plt.plot(xs, ys, linewidth=1)
   
    #September 
    ax2 = september.plot(linestyle='-')
    xs2 = np.arange(1979, 2016)
    ys2 = september_params[0] * xs + september_params[1] 
    plt.plot(xs2, ys2, linewidth=1)

def make_fig_2(get_Mar_Sept_frame):
    '''
    Creates a line graph for the March and Septmeber Anomaly data and fits a regression line to the data.

    Parameters:
        get_Mar_Sept_frame: dataframe with the mean and anomaly data
    Returns: none
    '''
    march_anom = pd.Series(get_Mar_Sept_frame['March_anomalies'])
    september_anom = pd.Series(get_Mar_Sept_frame['September_anomalies'])
    march_params = get_ols_parameters(march_anom)
    september_params = get_ols_parameters(september_anom)
    plt.title('The Anomaly', fontsize = 25)
    


    #March
    ax = march_anom.plot(linestyle='-')
    ax.set_ylabel(r"NH Sea Ice Extent ($10^6$ km$^2$)")
    ax.yaxis.label.set_fontsize(24)
    xs = np.arange(1979, 2016)
    ys = march_params[0] * xs + march_params[1] 
    plt.plot(xs, ys, linewidth=1)
   
    #September 
    ax2 = september_anom.plot(linestyle='-')
    xs2 = np.arange(1979, 2016)
    ys2 = september_params[0] * xs2 + september_params[1] 
    plt.plot(xs2, ys2, linewidth=1)



    
    




    

#==========================================================
def main():
    '''
    Using the function above, this reads and the data, calculates the means and anomalies for the March and Septmeber. 
    Gets the parameters from that data using OLS, and uses the parameters to make a prediction of when the Sea Ice will melt away for the winter and summer months. 
    Will create two graphs representing the mean, anolmaly and regression data for march and september.


    '''
    df = get_Mar_Sept_frame()
    march = pd.Series(df['March_means'])
    september = pd.Series(df['September_means'])
    march_anomaly = pd.Series(df['March_anomalies'])
    september_anomaly = pd.Series(df['September_anomalies'])

    march_p = get_ols_parameters(march)
    september_p = get_ols_parameters(september)
    march_anomaly_p = get_ols_parameters(march_anomaly)
    september_anomaly_p = get_ols_parameters(september_anomaly)

    make_prediction(march_p, description="Winter prediction",x_name='Time', y_name='Artic Sea Ice', ceiling = True)
    print()
    make_prediction(september_p, description="Summer prediction", x_name='Time', y_name='Artic Sea Ice', ceiling = True)
    
    make_fig_1(get_Mar_Sept_frame())
    plt.figure()
    make_fig_2(get_Mar_Sept_frame())
    plt.show()
        
if __name__ == '__main__':
    main()
