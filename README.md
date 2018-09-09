# Sea-Ice-Linear-Regression

Using March and September Artic Sea Ice data. This will use a simple linear regression to estimate when the artic sea ice will melt away completely

Using the functions within arcticSeaIce.py. This program reads in the data from data_79_16.csv, calculates the means and anomalies for the March and Septmeber, gets the parameters from the data using OLS, and uses the parameters to make a prediction of when the Sea Ice will melt away for the winter and summer months. 

Will create two graphs representing the mean, anolmaly and regression data for march and september.


## Expected Text Output
Winter prediction 2379 <br/>
76.0% of variation in Artic Sea Ice accounted for by Time (linear model) <br/>
Significance level of results: 0.0% <br/>
This result is statistically significant. <br/>

Summer prediction 2071 <br/>
75.0% of variation in Artic Sea Ice accounted for by Time (linear model) <br/>
Significance level of results: 0.0% <br/>
This result is statistically significant. <br/>