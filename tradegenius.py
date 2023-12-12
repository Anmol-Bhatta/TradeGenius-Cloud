#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Data Connect / process
from snowflake.snowpark.session import Session
import toml
import numpy as np
import pandas as pd
import sys

## Machine Learning
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima.arima import ARIMA
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import itertools


# In[30]:


##Data viz
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')

## Default settings
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[2]:


secrets = toml.load("secrets.toml")

accountname = secrets["SNOWFLAKE"]["account"]
user = secrets["SNOWFLAKE"]["user"]
password = secrets["SNOWFLAKE"]["password"]
role = secrets["SNOWFLAKE"]["role"]
database = secrets["SNOWFLAKE"]["database"]
schema = secrets["SNOWFLAKE"]["schema"]
warehouse = secrets["SNOWFLAKE"]["warehouse"]

connection_parameters = {
    "account": accountname,
    "user": user,
    "password": password,
    "role": role,
    "database": database,
    "schema": schema,
    "warehouse": warehouse,
    "ocsp_fail_open":"False"
}

session = Session.builder.configs(connection_parameters).create()


# In[3]:


print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())


# In[4]:


price_sdf = session.table('historical_prices')
print(type(price_sdf))


# In[5]:


price_df = price_sdf.to_pandas()
print(type(price_df))


# In[6]:


# Compare size
print('Size in MB of Snowpark DataFrame in Memory:', np.round(sys.getsizeof(price_sdf) / (1024.0**2), 2))
print('Size in MB of Pandas DataFrame in Memory:', np.round(sys.getsizeof(price_df) / (1024.0**2), 2))


# In[7]:


price_df


# In[8]:


price_df.shape


# In[9]:


price_df.isna().sum()


# In[10]:


price_df.duplicated().sum()


# In[11]:


price_df.drop_duplicates(subset='DATE', keep="last",inplace=True)
price_df.duplicated().sum()
price_df.shape


# In[12]:


price_df = price_df.sort_values(by='DATE',ignore_index=True)


# In[13]:


price_df.info()


# In[14]:


price_df['DATE'] = pd.to_datetime(price_df['DATE'])


# In[15]:


price_df['YEAR'] = price_df['DATE'].dt.year
price_df['MONTH'] = price_df['DATE'].dt.month
price_df['DAY'] = price_df['DATE'].dt.day
price_df['MONTHYEAR'] = price_df['DATE'].dt.strftime('%Y%m').astype(int)


# In[16]:


price_df.head()


# In[17]:


price_df.describe()


# In[33]:


trace = go.Scatter(x=price_df['DATE'], 
                   y=price_df['CLOSE'],
                   line_color='deepskyblue', 
                   name = 'Actual Prices')

data = [trace]
layout = dict(
    title='Daily Closing Price Google',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                    label='1m',
                    step='month',
                    stepmode='backward'),
                dict(count=3,
                    label='3m',
                    step='month',
                    stepmode='backward'),
                dict(count=6,
                    label='6m',
                    step='month',
                    stepmode='backward'),
                dict(count=12,
                    label='1Y',
                    step='month',
                    stepmode='backward'),
                dict(count=5,
                    label='5Y',
                    step='year',
                    stepmode='backward'),
                dict(step="all")
            ])
        ),
        title='Date',
        rangeslider=dict(
            visible = True
        ), type='date'
    ),
    yaxis=dict(title='Closing Price')
)
fig = dict(data=data, layout=layout)
py.iplot(fig);


# In[19]:


price_df.groupby(['DAY'])['CLOSE'].mean().plot.bar();


# In[20]:


price_df.groupby(['MONTH'])['CLOSE'].mean().plot.bar();


# In[21]:


price_df.groupby(['YEAR'])['CLOSE'].mean().plot.bar();


# In[32]:


corr = price_df.corr()
plt.figure(figsize=(18,12))
sns.heatmap(corr,annot=True);


# In[23]:


data = price_df[["CLOSE"]]
data.reset_index(inplace=True)
data.head()


# In[24]:


X = np.array(data[['index']])
y = np.array(data['CLOSE'])


# In[25]:


train_len = int(len(X)* 0.8)
X_train, X_test = X[:train_len], X[train_len:]
y_train, y_test = y[:train_len], y[train_len:]


# In[26]:


linear_model = LinearRegression()
linear_model.fit(X_train,y_train)


# In[27]:


y_pred = linear_model.predict(X_test)


# In[34]:


def plot_prediction(actual, prediction, title, y_label='Closing Price', x_label='Trading Days'):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Actual Price')
    plt.plot(prediction, '#0000FF', label='Predicted Price')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.show()  


# In[35]:


plot_prediction(y_test,y_pred,'Actual vs Prediction Linear Regression')


# In[36]:


def evaluate_model(y_true, y_pred):
    print(f"Root Mean squared error: {mean_squared_error(y_true , y_pred,  squared=False)}")
    print(f"Mean absolute error: {mean_absolute_error(y_true , y_pred)}")
    print(f"Mean absolute percentage error: {mean_absolute_percentage_error(y_true , y_pred)}")


# In[37]:


evaluate_model(y_test,y_pred)


# In[38]:


train_len = int(price_df.shape[0] * 0.8)
train_data, test_data = price_df[:train_len], price_df[train_len:]

y_train = train_data['CLOSE'].values
y_test = test_data['CLOSE'].values


# In[39]:


parima_model = ARIMA(order=(1,1,1))
parima_model.fit(y_train)


# In[40]:


y_parima_pred = parima_model.predict(len(y_test))


# In[41]:


plot_prediction(y_test, y_parima_pred,'Actual vs Prediction ARIMA')


# In[42]:


evaluate_model(y_test, y_parima_pred)


# In[43]:


data = price_df[["DATE","CLOSE"]] 
data.columns = ("ds","y") #renaming the columns of the dataset
data.head(5)


# In[44]:


train_len = int(data.shape[0] * 0.8)
train_data, test_data = data[:train_len], data[train_len:]


# In[45]:


prophet_model = Prophet() 
prophet_model.fit(train_data)


# In[46]:


future = test_data[['ds']]
#future = prophet_model.make_future_dataframe(periods=4,include_history=True) #we need to specify the number of days in future
prophed_pred_df = prophet_model.predict(future)
prophed_pred_df.tail()


# In[47]:


prophet_model.plot(prophed_pred_df)

plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show(fig);


# In[48]:


prophet_model.plot_components(prophed_pred_df);


# In[49]:


evaluate_model(test_data[['y']], prophed_pred_df[['yhat']])


# In[88]:


# Initiate the model
baseline_model = Prophet()
# Fit the model on the training dataset
baseline_model.fit(data)
# Cross validation
baseline_model_cv = cross_validation(model=baseline_model, initial='200 days', period='30 days', horizon = '30 days', parallel="processes")


# In[51]:


baseline_model_p = performance_metrics(baseline_model_cv, rolling_window=1)
baseline_model_p


# In[52]:


print("Baseline Prophet model MAPE :",baseline_model_p['mape'].values[0])


# In[53]:


print("Default value of changepoint_range :",prophet_model.changepoint_range)


# In[86]:


# Initiate the model
manual_model = Prophet(changepoint_range=0.99)
# Fit the model on the training dataset
manual_model.fit(data)
# Cross validation
manual_model_cv = cross_validation(manual_model, initial='200 days', period='30 days', horizon = '30 days', parallel="processes")
# Model performance metrics
manual_model_p = performance_metrics(manual_model_cv, rolling_window=1)


# In[57]:


print("Manual Prophet model MAPE :",manual_model_p['mape'].values[0])


# In[85]:


# Set up parameter grid
param_grid = {  
    'changepoint_prior_scale': [0.001, 0.05, 0.08, 0.5],
    'seasonality_prior_scale': [0.01, 1, 5, 10, 12],
    'seasonality_mode': ['additive', 'multiplicative'],
    'changepoint_range': [0.99,0.8]
}
# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
# Create a list to store MAPE values for each combination
mapes = [] 
# Use cross validation to evaluate all parameters
for params in all_params:
    # Fit a model using one parameter combination
    m = Prophet(**params).fit(data)  
    # Cross-validation
    df_cv = cross_validation(m, initial='200 days', period='30 days', horizon = '30 days', parallel="processes")
    # Model performance
    df_p = performance_metrics(df_cv, rolling_window=1)
    # Save model performance metrics
    mapes.append(df_p['mape'].values[0])
    
# Tuning results
tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = mapes
# Find the best parameters
best_params = all_params[np.argmin(mapes)]
print(best_params)


# In[59]:


best_params


# In[60]:


best_params = {'changepoint_prior_scale': 0.5, 
               'seasonality_prior_scale': 0.01, 
               'seasonality_mode': 'additive',
               'changepoint_range': 0.99
                  }
best_params


# In[61]:


# Fit the model using the best parameters
auto_model = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'], 
                     seasonality_prior_scale=best_params['seasonality_prior_scale'], 
                     seasonality_mode=best_params['seasonality_mode'],
                     changepoint_range=best_params['changepoint_range'] 
                    )
# Fit the model on the training dataset
auto_model.fit(data)
# Cross validation
auto_model_cv = cross_validation(auto_model, initial='200 days', period='30 days', horizon = '30 days', parallel="processes")
# Model performance metrics
auto_model_p = performance_metrics(auto_model_cv, rolling_window=1)


# In[62]:


print("Tunned Prophet model MAPE :",auto_model_p['mape'].values[0])


# In[63]:


future = auto_model.make_future_dataframe(periods=5,include_history=False) 
future


# In[65]:


future = auto_model.make_future_dataframe(periods=180,
                                          include_history=True) 
prediction = auto_model.predict(future)


# In[ ]:


trace0 = go.Scatter(x=price_df['DATE'], y=price_df['CLOSE'],line_color='deepskyblue', name='Actual Prices')

trace1 = go.Scatter(x=prediction['ds'], y=prediction['yhat'],line_color='lime', name='Predicted Prices')

data = [trace0, trace1]
layout = dict(
    title='Actual Prices vs Predicted Prices',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                    label='1m',
                    step='month',
                    stepmode='backward'),
                dict(count=3,
                    label='3m',
                    step='month',
                    stepmode='backward'),
                dict(count=6,
                    label='6m',
                    step='month',
                    stepmode='backward'),
                dict(count=12,
                    label='1Y',
                    step='month',                     
                     stepmode='backward'),
                dict(count=5,
                    label='5Y',
                    step='year',
                    stepmode='backward'),
                dict(step="all")
            ])
        ),
        title='Date',
        rangeslider=dict(
            visible = True
        ), type='date'
    ),
    yaxis=dict(title='Closing Price')
)
fig = dict(data=data, layout=layout)
py.iplot(fig);


# In[68]:


from snowflake.snowpark.session import Session
import snowflake.snowpark.types as T
import toml
import pandas as pd
from prophet import Prophet
import json


# In[69]:


accountname = secrets["SNOWFLAKE"]["account"]
user = secrets["SNOWFLAKE"]["user"]
password = secrets["SNOWFLAKE"]["password"]
role = secrets["SNOWFLAKE"]["role"]
database = secrets["SNOWFLAKE"]["database"]
schema = secrets["SNOWFLAKE"]["schema"]
warehouse = secrets["SNOWFLAKE"]["warehouse"]

connection_parameters = {
    "account": accountname,
    "user": user,
    "password": password,
    "role": role,
    "database": database,
    "schema": schema,
    "warehouse": warehouse,
    "ocsp_fail_open":"False"
}

session = Session.builder.configs(connection_parameters).create()


# In[70]:


print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())


# In[71]:


session.sql('CREATE OR REPLACE STAGE ML_MODELS').collect()


# In[72]:


pd.DataFrame(session.sql('SHOW STAGES').collect())


# In[84]:


def sproc_predict_using_prophet(session: Session, 
                                training_table: str,
                                include_history: str,
                                period: int) -> T.Variant:
    
    # Loading data into pandas dataframe
    data_sdf = session.table(training_table)    
    data = data_sdf.select('DATE','CLOSE').to_pandas()
    data.drop_duplicates(subset='DATE', keep="last",inplace=True)
    data.sort_values(by='DATE',inplace=True)
    data.columns = ['ds','y']
    
    # Actual model training
    from prophet import Prophet
     
    model = Prophet(changepoint_prior_scale=0.5,
                    seasonality_prior_scale=0.01,
                    seasonality_mode='additive',
                    changepoint_range=0.99
                       )
    model.fit(data)
    
    if include_history == 'Y':
        flag = True
    else:
        flag = False
        
    future_df = model.make_future_dataframe(periods=period,
                                           include_history=flag)
    forecast = model.predict(future_df)
  
    return forecast.to_dict()


# In[80]:


#Adding paclaeges to session.
session.add_packages('snowflake-snowpark-python','prophet')
# Registering the function as a Stored Procedure
sproc_predict_using_prophet = session.sproc.register(func=sproc_predict_using_prophet, 
                                            name='sproc_predict_using_prophet', 
                                            is_permanent=True, 
                                            replace=True, 
                                            stage_location='@ML_MODELS', 
                                            packages=['prophet==1.0.1', 'holidays==0.18','snowflake-snowpark-python'])


# In[81]:


training_table = 'historical_prices'
show_history='N'
future_days=4

pred_list = session.sql(
            "call sproc_predict_using_prophet('{}', '{}',{})".format(training_table,show_history, future_days)   
            ).collect()

pred_df = pd.DataFrame(json.loads(pred_list[0][0]))
pred_df = pred_df[['ds','yhat']]
pred_df['ds'] = pd.to_datetime(pred_df['ds']).dt.date
pred_df.columns = ['DATE', 'PRICE']
pred_df


# In[82]:


training_table = 'historical_prices'
show_history='N'
future_days=4

pred_list = session.sql(
            "call sproc_predict_using_prophet('{}', '{}',{})".format(training_table,show_history, future_days)   
            ).collect()

pred_df = pd.DataFrame(json.loads(pred_list[0][0]))
pred_df = pred_df[['ds','yhat']]
pred_df['ds'] = pd.to_datetime(pred_df['ds']).dt.date
pred_df.columns = ['DATE', 'PRICE']
pred_df


# In[83]:


import toml
import os
directory = os.path.join(os.getcwd() + "/.streamlit")
if not os.path.exists(directory):
    os.makedirs(directory)
    
secrets = toml.load("secrets.toml")

output_file_name = os.path.join(directory+"/secrets.toml")
with open(output_file_name, "w") as toml_file:
    toml.dump(secrets, toml_file)


# In[ ]:




