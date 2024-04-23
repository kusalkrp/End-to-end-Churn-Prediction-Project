import time
import pandas as pd
import streamlit as st
import requests
import json
import plotly.graph_objects as go

# Customer Insights
def customer_analytics(df):
    col1, col2, col3 = st.columns(3)
    contract_details = df['contract'].value_counts()
    internet_details = df['internet_service'].value_counts()
    payment_details = df['payment_method'].value_counts()

    with col1:
        labels = contract_details.index.tolist()
        values = contract_details.values.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
        fig.update_layout(title_text="<b>Contract Type</b>")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        labels = internet_details.index.tolist()
        values = internet_details.values.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
        fig.update_layout(title_text="<b>Internet Service</b>")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        labels = payment_details.index.tolist()
        values = payment_details.values.tolist()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
        fig.update_layout(title_text="<b>Payment Method</b>")
        st.plotly_chart(fig, use_container_width=True)


# Sidebar for selecting section
section = st.sidebar.radio("Select Section", ["Single Customer Prediction", "Batch Customer Prediction"])

if section == "Single Customer Prediction":
    st.title('Single Customer Churn Prediction')
    

    # Input Form
    st.sidebar.header('Input Features')
    
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'], index=0)
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ['Yes', 'No'], index=0)
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'], index=0)
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'], index=0)
    tenure = st.sidebar.number_input('Tenure', min_value=0, max_value=100, value=None)
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'], index=0)
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No'], index=0)
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'], index=0)
    online_security = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet service'], index=0)
    online_backup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet service'], index=0)
    device_protection = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet service'], index=0)
    tech_support = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet service'], index=0)
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'], index=0)
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'], index=0)
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'], index=0)
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'], index=0)
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], index=0)
    monthly_charges = st.sidebar.number_input('Monthly Charges', min_value=0.0, max_value=5000.0, value=None)
    total_charges = st.sidebar.number_input('Total Charges', min_value=0.0, max_value=100000.0, value=None)

    # Prediction Button
    if st.sidebar.button('Predict'):
        # Create data dictionary
        data = {
            'gender': gender,
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'tenure': tenure if tenure is not None else 0,
            'phone_service': phone_service,
            'multiple_lines': multiple_lines,
            'internet_service': internet_service,
            'online_security': online_security,
            'online_backup': online_backup,
            'device_protection': device_protection,
            'tech_support': tech_support,
            'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies,
            'contract': contract,
            'paperless_billing': paperless_billing,
            'payment_method': payment_method,
            'monthly_charges': monthly_charges if monthly_charges is not None else 0.0,
            'total_charges': total_charges if total_charges is not None else 0.0,
        }

        # API endpoint URL
        url = 'http://127.0.0.1:8000/predict/'

        try:
            # Make POST request to API
            response = requests.post(url, json=data)

            # Check if request was successful
            if response.status_code == 200:
                # Parse JSON response
                result = response.json()

                # Display prediction
                st.write(f"Predicted Churn: {result.get('Predicted_Churn', 'Prediction not available')}")

                # Display retention message if available
                retention_message = result.get('Retention_Message')
                if retention_message:
                    st.write("Retention Message:")
                    st.write(retention_message)
                else:
                    st.write("Retention Message not available.")
            else:
                st.write(f"Error: {response.status_code} - {response.text}")

        except requests.RequestException as e:
            st.write(f"Error making API request: {e}")

elif section == "Batch Customer Prediction":
    st.title('Batch Customer Churn Prediction')
    

    # Upload CSV file
    upload_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if upload_file is not None:
        # Read uploaded CSV file
        df = pd.read_csv(upload_file)
        
        # Predict Batch Button
        if st.sidebar.button('Predict Batch'):
            # Prepare data for API request
            batch_data = {
                "records": df.to_dict(orient="records")
            }

            # API endpoint URL
            url = 'http://127.0.0.1:8000/predict_batch/'

            try:
                # Make POST request to API
                with st.spinner('Predicting churn...'):
                    progress_bar = st.progress(0)
                    
                    response = requests.post(url, json=batch_data)
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        # Parse JSON response
                        result = response.json()

                        # Add Predictions column to DataFrame
                        df.insert(loc=0, column='Predictions', value=result['predictions'])
                        
                        # Display uploaded data with Predictions column
                        st.subheader('Uploaded Data with Predictions')
                        st.write(df)

                        # Add padding and horizontal line
                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown('<p style="margin-top: 20px;"></p>', unsafe_allow_html=True)

                        # Analysis
                        num_customers = len(df)
                        num_churning = df['Predictions'].value_counts().get('The customer is likely to churn', 0)
                        num_staying = df['Predictions'].value_counts().get('The customer is staying', 0)

                        # Display analysis
                        st.subheader('Analysis')
                        st.write(f"Number of Customers: {num_customers}")
                        st.write(f"Number of Churning Customers: {num_churning}")
                        st.write(f"Number of Staying Customers: {num_staying}")

                        # Add padding and horizontal line
                        st.markdown('<hr>', unsafe_allow_html=True)
                        st.markdown('<p style="margin-top: 20px;"></p>', unsafe_allow_html=True)

                        # Display customer analytics
                        customer_analytics(df)
                        
                    else:
                        st.write(f"Error: {response.status_code} - {response.text}")

                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)

            except requests.RequestException as e:
                st.write(f"Error making API request: {e}")