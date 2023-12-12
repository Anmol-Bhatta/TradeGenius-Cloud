#Import Required Libraries 
import streamlit as st
from snowflake.snowpark.session import Session
from plotly import graph_objs as go
import pandas as pd
import json
#import toml

#Extract credentials from secret file 
#secrets = toml.load("secrets.toml")
accountname = st.secrets["SNOWFLAKE"]["account"]
user = st.secrets["SNOWFLAKE"]["user"]
password = st.secrets["SNOWFLAKE"]["password"]
role = st.secrets["SNOWFLAKE"]["role"]
database = st.secrets["SNOWFLAKE"]["database"]
schema = st.secrets["SNOWFLAKE"]["schema"]
warehouse = st.secrets["SNOWFLAKE"]["warehouse"]

#Setting Menu options and web page configurations 
st.set_page_config(
     page_title="Stock Price Prediction",
     #layout="wide",
     page_icon="💹",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.vinoddhole.com',
         'About': "**Stock Price Prediction** App powered by AWS, Snowflake, Python, Snowpark and Streamlit"
     }
 )

# Adding SideBar
st.sidebar.title("Stock Price Prediction")
st.sidebar.markdown("**Stock Price Prediction** App powered by AWS, Snowflake, Python, Snowpark and Streamlit")
st.sidebar.markdown("Author: [Vinod Dhole](https://medium.com/@vinodvidhole)")
st.sidebar.markdown('''
Step by Step Guide to build this app : [Part 1](https://medium.com/@vinodvidhole/c304a8b3e319)
,[Part 2](https://medium.com/@vinodvidhole/3f20763ac7a6)
,[Part 3](https://medium.com/@vinodvidhole/98b8cea5831c)
'''
)
st.sidebar.markdown("Source: [Github](https://github.com/vinodvidhole/stockprice-predictions)")
st.sidebar.markdown("Disclaimer: The purpose of this app is only to demonstrate machine learning lifecycle using multiple techniques & tools like Snowflake, AWS, Python and Streamlit. This is not investment advice.")

# Create Session object
def create_session_object():
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

    #print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
    return session

#Extract the data from SnowFlake load on pandas DataFrame 
def fetch_data(session):
    df = session.table('historical_prices').to_pandas()
    df.drop_duplicates(subset='DATE', keep="last",inplace=True)
    df = df.sort_values(by='DATE',ascending=False)
    return df
    
#Setting web page Title
st.title('Stock Price Prediction')

#Drop down selection for Ticker 
#Currently there is Only one option Google, can add more options like stocks = ('GOOG','AAPL')
stocks = ('GOOG',)
selected_stock = st.selectbox('Select Ticker', stocks)


if __name__ == "__main__":

    #Function call to connect to SnowFlake
    session = create_session_object()

    #Setting status progression text in webpage 
    data_load_state = st.text('Loading data...')
    
    #Function call to Get data 
    price_df = fetch_data(session)
    
    #Setting process completion text in webpage 
    data_load_state.text('Loading data... done!')

    st.subheader('Historical Prices')
    
    #Displaying Historical pricing data 
    st.dataframe(price_df)
       
    st.subheader('Historical Price Trend')

    #Creating Visualization for Historical Prices (This is exactly same code from Article 2)
    trace = go.Scatter(x=price_df['DATE'], y=price_df['CLOSE'],line_color='deepskyblue', name = 'Actual Prices')

    data = [trace]
    layout = dict(
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
    st.plotly_chart(fig)

    with st.expander("Click me for Help!"):
        st.write("""
            Use the slidebar to choose the number of future days to predict.

            Select (Yes/No) option to include historical dates for predictions
            """)

    #Slider bar for users to select no. future days Range 1 - 1825 days, default is 180 days 
    days = st.slider('Select No. of days for prediction', 1, 1852,180)
    
    #Yes/No Selection option to show Historical Prices
    show_history_option = st.selectbox(
        "Show Historical Predicted Prices?",
        ("No", "Yes")
    )
    if show_history_option=='Yes':
        show_history = 'Y'
    else:
        show_history = 'N'

    #On Button click run code inside if statement 
    if st.button('Forecast Predictions'):
        
        prediction_state = st.text('Predicting Future Prices...')

        #Calling snowflake stored procedure "sproc_predict_using_prophet(table,show_hist, days", which will predict future prices (Same code grabbed from Article 3)
        pred_list = session.sql(
                "call sproc_predict_using_prophet('{}', '{}',{})".format('historical_prices',show_history, days)   
                ).collect()

        #Load the Prediction data from stored procedure into DataFrame 
        pred_df = pd.DataFrame(json.loads(pred_list[0][0]))
        pred_df = pred_df[['ds','yhat']]
        pred_df['ds'] = pd.to_datetime(pred_df['ds']).dt.date
        pred_df.columns = ['DATE', 'CLOSE']
        pred_df.sort_values(by='DATE',inplace=True)
        
        st.subheader('Predicted Prices')

        #Display the prediction output 
        st.dataframe(pred_df)
        st.subheader('Predicted Price Trend')

        trace0 = go.Scatter(x=price_df['DATE'], y=price_df['CLOSE'],line_color='deepskyblue', name='Actual Prices')

        trace1 = go.Scatter(x=pred_df['DATE'], y=pred_df['CLOSE'],line_color='lime', name='Predicted Prices')

        #Visualization of Actual Prices vs Predicted Prices (This is exactly same code from Article 2)
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
        st.plotly_chart(fig)
        prediction_state.text('Prediction Done!')
