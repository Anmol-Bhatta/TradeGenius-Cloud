# Stock Price Prediction Application

This project is a cloud-based stock price prediction application that utilizes AWS, Snowflake, Python, and Streamlit. The application automates the collection of historical stock prices, processes the data, and provides real-time predictions through a web interface. 

## Project Overview

The project follows these key steps:
1. **Data Collection:** Uses AWS Lambda to automate the daily retrieval of historical stock price data and stores it in an AWS S3 bucket.
2. **Data Storage:** Integrates with Snowflake using Snowpipe to ingest the data in real-time from the S3 bucket.
3. **Data Processing:** Extracts and preprocesses data using Python, performing exploratory data analysis (EDA) to prepare for modeling.
4. **Machine Learning:** Develops a machine learning model for stock price prediction and deploys it on Snowflake.
5. **Web Application:** Builds an interactive web app using Streamlit for displaying real-time stock price predictions.

## Technologies Used

- **AWS:** Lambda, S3 for data collection and storage.
- **Snowflake:** Data warehouse for real-time data ingestion and storage.
- **Python:** For data extraction, preprocessing, and model development.
- **Snowpipe:** Integration for automating data ingestion into Snowflake.
- **Streamlit:** Web application for real-time stock predictions.

## Getting Started

### Prerequisites

- An AWS account with access to S3 and Lambda services.
- A Snowflake account for data warehousing.
- Python 3.9 or above installed locally.
- Streamlit for web app development (`pip install streamlit`).

### Steps to Recreate

1. **Data Collection:**
   - Set up an AWS Lambda function to fetch historical stock prices (e.g., using `yfinance` library).
   - Store retrieved data in CSV format in an AWS S3 bucket.

2. **Data Storage:**
   - Integrate AWS S3 with Snowflake using Snowpipe for real-time data ingestion.
   - Configure Snowflake storage integration and IAM roles for secure data access.

3. **Data Processing:**
   - Use Python to extract, preprocess, and perform exploratory data analysis on the ingested data in Snowflake.

4. **Model Development:**
   - Develop and train a machine learning model for stock price prediction.
   - Deploy the model on Snowflake for scalable prediction tasks.

5. **Web Application:**
   - Build a Streamlit web application to display real-time predictions.
   - Deploy the web app on Streamlit Cloud for easy access.

## Repository Structure

