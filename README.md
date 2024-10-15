# Customer-Churn-Prediction-with-ML

This project focuses on predicting customer churn using various machine learning algorithms. The dataset used is the Telco Customer Churn dataset, which includes customer information along with whether they have churned or not. The goal is to build a model that predicts customer churn based on features such as customer tenure, monthly charges, and services used.
The dataset contains the following columns:

	•	gender, SeniorCitizen, Partner, Dependents: Customer demographic information.
	•	tenure, PhoneService, MultipleLines, InternetService, etc.: Information about the services each customer uses.
	•	MonthlyCharges, TotalCharges: The monetary charges for each customer.
	•	Churn: The target variable indicating whether a customer churned or not.

Dataset Source: Telco Customer Churn Dataset
Project Steps

Exploratory Data Analysis (EDA)

	•	Initial Data Overview: Checking for missing values, class imbalance, and understanding the general structure of the dataset.
	•	Numerical and Categorical Variables: Visualizing the distribution of numerical features (histograms, boxplots) and analyzing categorical features (countplots).

Data Preprocessing

	•	Missing Data Handling: Missing values in TotalCharges were replaced with the median of the column.
	•	Label Encoding: Applied label encoding to binary categorical columns such as gender, Churn, etc.
	•	One-Hot Encoding: Applied one-hot encoding to multi-class categorical variables such as InternetService, Contract, PaymentMethod.

Feature Engineering

	•	Created new features:
	•	High_MonthlyCharge: A binary feature indicating if the monthly charge is higher than the average.
	•	High_TotalCharge: A binary feature indicating if the total charge is higher than the average.
	•	High_Service_Usage: A feature capturing the number of additional services (e.g., OnlineSecurity, StreamingTV) the customer uses.
	•	Tenure_Churn_Interaction: An interaction feature between tenure and Churn.
	•	Age_Group: Categorized customers into Senior or Non-Senior based on the SeniorCitizen column.

Model Building

Several machine learning algorithms were applied, including:

	•	Logistic Regression
	•	Decision Tree Classifier
	•	Random Forest Classifier
	•	K-Nearest Neighbors (KNN)
	•	Support Vector Classifier (SVC)

The dataset was split into training and testing sets, and the models were trained using the training set and evaluated on the test set.

Model Evaluation

	•	Confusion Matrix and Classification Report: Evaluated model performance based on accuracy, precision, recall, and F1-score.
	•	ROC Curve and AUC Score: Used to assess the model’s ability to distinguish between churn and non-churn customers.

Conclusion

This project demonstrates the complete process of building a customer churn prediction model, starting from exploratory data analysis and preprocessing, to feature engineering and model building. The results of different machine learning algorithms were compared, and the best performing model can be used to predict churn in real-world scenarios.

Contributing

If you’d like to contribute to this project, feel free to submit a pull request or open an issue.
