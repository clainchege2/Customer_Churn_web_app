import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from utils import create_gauge_chart 

client = OpenAI(
  base_url = 'https://api.groq.com/openai/v1',
  api_key = os.environ.get('GROQ_API_KEY')
)

def explain_prediction(probability, input_dict, surname):
  prompt = f""" Youare an expert data scientist at a bank, where you specialize in predicting the probability of a person having a credit card debt. You are given the following information:
  Your machine learning model has predicted that a customer name {surname} has a probability of {round(probability * 100,1)}% probability of churning, based on the following:
  Here is the customer profile:
    {input_dict}
    Herer are the machine learning models top 10 most important features for predicting churn:
    
    Feature	Importance_XGB
    ------------------------------
    4	NumOfProducts	    |  0.323888	
    6	IsActiveMember	  |  0.164146	
    1	Age	              |  0.109550	
    9	Geography_Germany	|  0.091373	
    3	Balance	0.052786	|  0.139153
    8	Geography_France	|  0.046463	
    11	Gender_Female	  |  0.045283	
    10	Geography_Spain	|  0.036855	
    0	CreditScore	      |  0.035005	
    7	EstimatedSalary	  |  0.032655	
    5	HasCrCard	        |  0.031940	
    2	Tenure	          |  0.030054	
    12	Gender_Male	    |  0.000000	


    {pd.set_option('display.max_columns', None)}

    Here are summary statistics for churned customers:
    {df[df.Exited == 0].describe()}


  - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why   they are at risk of churning.
  - If the  customer has less than a 40% risk of churning, generate a 3 sentence explanation   of why they are not at risk of churn.
  - Your explanation should be based on the customer's profle and the machine learning model's top 10 most important features.


  -Don't mention the probability of churning, or the machine learning model, or say anything     like "Based on the machine learning model's prediction and top 10 most important features",   just explain the prediction.
  -Also don't show the promt lenghth, just explain the prediction. 
  - Create the profile stats as bullet points.
  only show the country with label 1 on geography features.
  """

  print('EXPLANATION PROMPT', prompt)

  raw_response = client.chat.completions.create(
    model='llama-3.2-3b-preview',
    messages=[{
      "role": "user",
      "content": prompt
    }]
  )
  return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
  prompt = f"""You are a manager at Chase bank. You are responsible for the customer experience at the bank and ensuring customers stay with the bank and are incentivized with various offers.
  You notice the customer's information:
    {input_dict}


    Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.
    Make sure to list out a set of incentives to saty based on their information, in bullet point formart. Don't ever mention the probability of churning, or the machine learning model to the customer
    The customer is already aware of their tenure, dont mention thier statistics.
    For the offers, find real offers that are out there.
    Use official formats, dont use any slang or jargon.
  """
  raw_response = client.chat.completions.create(
    model='llama-3.1-8b-instant',
    messages=[{
      "role": "user",
      "content": prompt
    }],
  )
  print('\n\nEmail PROMT', prompt)
  return raw_response.choices[0].message.content
  
def load_model(filename):
  with open(filename, 'rb') as file:
    return pickle.load(file)

xgboost_model = load_model('best_xgboost_model.pkl')
decision_tree_model = load_model('decision_tree.pkl')
naive_bayes_model = load_model('naive_bayes.pkl')
random_forest_model = load_model('random_forest.pkl')
voting_classifier_model = load_model('voting_classifier_soft.pkl')
Knn_model = load_model('knn.pkl')
logistic_regression_model = load_model('logistic_regression.pkl')

def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
  input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,  
    'NumOfProducts': num_products,
    'HasCrCard': int(has_credit_card),
    'IsActiveMember': int(is_active_member),
    'EstimatedSalary': estimated_salary,
    'Geography_France': 1 if location == 'France' else 0,
    'Geography_Germany':1 if location == 'Germany' else 0,
    'Geography_Spain':1 if location == 'Spain' else 0,
    'Gender_Female':1 if gender == 'Female' else 0,
    'Gender_Male':1 if gender == 'Male' else 0
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict

def make_predictions(input_df, input_dict):
  probabilities = {
    'Decision Tree': decision_tree_model.predict_proba(input_df)[0][1],
    'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
    'K Nearest Neighbors': Knn_model.predict_proba(input_df)[0][1],
    'Naive Bayes': naive_bayes_model.predict_proba(input_df)[0][1],
    'Logistic Ression': logistic_regression_model.predict_proba(input_df)[0][1],
    'Voting Classifier': voting_classifier_model.predict_proba(input_df)[0][1],
    'XGBoost': xgboost_model.predict_proba(input_df)[0][1]
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1 = st.container()

  with col1:
    fig = create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f'The customer has a {avg_probability:.2%} probability of churning.')
    
     

  st.markdown('### Model Probabilities')
  for model, prob in probabilities.items():
    st.write(f'{model}: {prob}')
  st.write(f'##### Average Probability: {avg_probability}')

  return avg_probability




st.title("Customer Churn Prediction")

df =  pd.read_csv('churn.csv')

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows() ]

selected_customer_option = st.selectbox("Select a customer", customers)


if selected_customer_option:

  selected_customer_id = int(selected_customer_option.split(" - ")[0])

  print('Selected Customer ID:', selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]

  print('Surname:', selected_surname)

  selected_customer = df[df['CustomerId'] == selected_customer_id].iloc[0]  

  print('Selected Customer Row:', selected_customer)


  col1, col2 = st.columns(2)
  
  with col1:
    credit_score = st.number_input(
      'Credit SCore',
      min_value=300,
      max_value=850,
      value=int(selected_customer['CreditScore']))
    
    location = st.selectbox(
      'Location', ['Spain', 'France', 'Germany'],
      index=['Spain', 'France', 'Germany'].index(selected_customer['Geography']))

    gender = st.radio("Gender", ['Male', 'Female'],
        index=0 if selected_customer['Gender'] == 'Male' else 1)

    age = st.number_input('Age', 
    min_value=18,
    max_value=100,
    value=int(selected_customer['Age']))


    tenure = st.number_input(
      'Tenure (years)',
      min_value=0,
      max_value=50,
      value=int(selected_customer['Tenure']))
    
  with col2:
    balance = st.number_input(
      'Balance',
      min_value=0.0,
      value=float(selected_customer['Balance']))

    num_products = st.number_input(
      'Number of Products Purchased',
      min_value=1,
      value=int(selected_customer['NumOfProducts']))

    has_credit_card = st.checkbox(
      'Has Credit Card',
      value=bool(selected_customer['HasCrCard']))
    
    is_active_member = st.checkbox(
      'Is Active Member',
      value=bool(selected_customer['IsActiveMember']))
    
    estimated_salary = st.number_input(
      'Estimated Salary',
      min_value=0.0,
      value=float(selected_customer['EstimatedSalary']))
    
  input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)


avg_probability = make_predictions(input_df, input_dict)

explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

st.markdown('---')
st.markdown('Explanation of Prediction')
st.markdown(explanation)


email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])  

st.markdown('---')
st.subheader('Email to Customer')
st.markdown(email)