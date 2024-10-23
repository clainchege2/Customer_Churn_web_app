This project predicts customer churn probability using pre-trained machine learning models and generates personalized emails with retention incentives based on customer profiles. The application is built using Python, Streamlit for the user interface, OpenAI for generating explanations and email content, and Retrieval-Augmented Generation (RAG) to enhance email personalization.

**Key Features:**
Churn Prediction Models: The project uses several pre-trained models, including XGBoost, Decision Tree, Naive Bayes, Random Forest, K-Nearest Neighbors, Logistic Regression, and a Voting Classifier. These models, trained in the **ML notebook**, predict the likelihood of customer churn based on input features like credit score, balance, age, and activity level.

**Prediction Explanation:** Once a prediction is made, OpenAI is utilized to generate human-readable explanations for why a customer is at risk (or not) of churning. The explanation is based on the top 10 most important features from the model, such as number of products, age, and activity status.

**Incentive Email Generation (RAG):** For customers at risk of churning, the tool generates personalized emails offering retention incentives. Retrieval-Augmented Generation (RAG) is implemented to fetch real-world offers and use this information to enhance the email content. This allows for more accurate and relevant retention offers based on customer profiles.

**How It Works:**
Data Input: Users provide customer data, such as credit score, location, age, balance, and more.
Churn Prediction: The input is fed into pre-trained machine learning models (trained in the ML notebook) to predict the likelihood of churn.
Explanations: An explanation is generated for the prediction, detailing key features influencing the outcome.
RAG-Based Email Generation: For customers at risk of churn, the tool uses Retrieval-Augmented Generation to fetch real-world offers and generate a personalized retention email.
Models Used:
XGBoost
Decision Tree
Naive Bayes
Random Forest
K-Nearest Neighbors (KNN)
Logistic Regression
Voting Classifier (Soft)
Technologies:
**Python**: For data processing and machine learning model implementation.
**Streamlit**: To create an interactive web application.
**OpenAI API**: For generating natural language explanations and personalized email content.
**RAG** (Retrieval-Augmented Generation): To fetch and integrate real-time incentives into email generation.
**Pickle**: For saving and loading machine learning models.

**How to run the files.**
1. Open the Model_training Notebook and run all cells to eplore the dataset and create the ML models.
2. Next go to replit.com and upload both the Replit_notebook and utils.py file.
  * Makesure to install the streamlit template to run the files.
 
3. Press run to preview your code as web app.
  
4. Enjoy and edit code as you see fit, Thanks.

