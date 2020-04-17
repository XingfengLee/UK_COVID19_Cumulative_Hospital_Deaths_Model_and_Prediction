# Import the required Python libraries.
import numpy as np 
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

nDegree = 6 # degree of polynomials 
n       = 4 # number of days to predict

today   = dt.date.today()
tDate   = today.strftime("%b-%d-%Y")

# Data from the following website:
dStr    = 'https://coronavirus.data.gov.uk'
# Use pandas to read data from csv file 'coronavirus-deaths.csv' which is downloaded from dStr.
df      = pd.read_csv('coronavirus-deaths.csv',index_col=0)
data    = df.filter(like='United Kingdom', axis=0)
# data    = df.filter(like='England', axis=0)
yVal    = data['Cumulative hospital deaths']
da      = data['Reporting date']
y       = yVal.values[::-1]
date    = da.values[::-1]
# Convert datetime string to datetime object.
N       = len(date)
days    = []
for i in range(N):
	tmp = dt.datetime.strptime(date[i], '%m/%d/%Y')
	days.append(tmp)

# Generate the x-matrix of independent variables.
x       = np.arange(1,N+1)
x       = x[:, np.newaxis]

# Model the curve with polynomial series.
polynomial_features = PolynomialFeatures(degree=nDegree)
x_poly  = polynomial_features.fit_transform(x) # independent polynomial variables

model   = LinearRegression()# statistical linear regression model
model.fit(x_poly, y)# Fit data y with the polynomial function.
yFit    = model.predict(x_poly) # prediction with polynomials
mse     = np.sqrt(mean_squared_error(y,yFit))# mean square error of the fit
r2      = r2_score(y,yFit)# goodness of fit with r squared measure
print(mse)
print(r2)

# Predict the possible deaths yet to come.
n1          = n + 2
nDay        = N + n1 - 1 # Number of days to predict 
x_old       = np.arange(1,nDay)# Number of days from the first death from the spreadsheet to the last day of prediction
x_new       = x_old[:, np.newaxis]# independent variable with a future date
x_poly_new  = polynomial_features.fit_transform(x_new)# polynomial base
y_poly_pred = model.predict(x_poly_new)#future prediction of the deaths

# Construct date time for the prediction. 
now         = days[N-1] 
then        = now + dt.timedelta(days=n1-1)
days2       = mdates.num2date(mdates.drange(now,then,dt.timedelta(days=1) ) )
days3       = days[0:N-1] + days2 

# Plot prediction time and data.
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.plot(days3,y_poly_pred, 'bs-', label='line 1', linewidth=2)

# Plot the original data.
y2          = np.zeros(x_old.shape)
y2[0:N]     = y
y2[N:nDay]  = None
plt.plot(days3,y2, 'ro-', label='line 1', linewidth=2)
plt.grid()
plt.ylabel("Total Number of Cumulative Hospital Deaths (Y)", fontsize=10)
plt.title('UK COVID-19 Number of Deaths Prediction on ' + tDate)  
plt.gcf().autofmt_xdate() # Rotate the x-axis datetime text

plt.text(days[1],yFit[nDay-10-n] , 'Goodness-of-fit: R^2 = ' + str(int(np.round(r2,3)))) 
plt.text(days[1],yFit[nDay-15-n] ,  "MSE = "  + str(int(np.round(mse,3)))) 

plt.text(days[1],y[2] + 12000,'Data from GOV web:  ' + dStr, fontsize=10)
plt.legend(('Prediction', 'Original Data'))
plt.show()

