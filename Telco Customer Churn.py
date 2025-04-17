##########################################################################
# Telco Churn Prediction
# -CASE STUDY I-
##########################################################################

# BUSINESS PROBLEM:
# It is expected to develop a machine learning model that can predict customers who are likely to leave the company.  
# Telco customer churn data provides information about 7043 customers of a fictional telecom company  
# that provides home phone and internet services in California during the third quarter.  
# It shows which customers stayed, left, or signed up for the service.

# CustomerId: Customer ID  
# Gender: Gender  
# SeniorCitizen: Whether the customer is a senior citizen (1 = Yes, 0 = No)  
# Partner: Whether the customer has a partner (Yes, No)  
# Dependents: Whether the customer has dependents (Yes, No)  
# tenure: Number of months the customer has stayed with the company  
# PhoneService: Whether the customer has phone service (Yes, No)  
# MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)  
# InternetService: Customer’s internet service provider (DSL, Fiber optic, No)  
# OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)  
# OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)  
# DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)  
# TechSupport: Whether the customer has tech support (Yes, No, No internet service)  
# StreamingTV: Whether the customer has streaming TV service (Yes, No, No internet service)  
# StreamingMovies: Whether the customer has streaming movies service (Yes, No, No internet service)  
# Contract: Customer’s contract term (Month-to-month, One year, Two year)  
# PaperlessBilling: Whether the customer uses paperless billing (Yes, No)  
# PaymentMethod: Customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))  
# MonthlyCharges: The amount charged to the customer monthly  
# TotalCharges: The total amount charged to the customer  
# Churn: Whether the customer has churned (Yes or No)
##########################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import train_test_split, cross_validate


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Calculating threshold values for variables in the dataset
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Checking whether there are any outliers in the variables of the dataset
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Replacing outlier values in the variable with the calculated threshold values
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


###############################################################################
# Task 1: Exploratory Data Analysis
###############################################################################

#########################
# Step 1: Identify numerical and categorical variables
#########################

def load():  
    data = pd.read_csv("hafta6_machinelearning/case1/Telco-Customer-Churn.csv")
    return data


df = load()

df.head()
df.shape  # (7043, 21)


def check_df(dataframe, head=5):
    print("##################### Shape #######################")
    print(dataframe.shape)
    print("\n##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Nunique #####################")
    print(dataframe.nunique())
    print("\n##################### Head ######################")
    print(dataframe.head())
    print("\n##################### Tail ######################")
    print(dataframe.tail())
    print("\n##################### NA ########################")
    print(dataframe.isnull().sum())
    print("\n################### Describe ####################")
    print(dataframe.describe().T)
    print("\n################### Quantiles ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numerical, and categorical but cardinal variables in the dataset.

    Note:
        Categorical variables also include numerical-looking categorical variables 
        (e.g., Survived, Pclass).

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe from which variable names will be extracted.
    cat_th : int, optional
        Class threshold for numerical but categorical variables.
    car_th : int, optional
        Class threshold for categorical but cardinal variables (e.g., Name, Ticket).

    Returns
    -------
    cat_cols : list
        List of categorical variables
    num_cols : list
        List of numerical variables
    cat_but_car : list
        List of categorical-looking but cardinal variables

    Examples
    --------
    import seaborn as sns  
    df = sns.load_dataset("iris")  
    print(grab_col_names(df))

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of variables  
    num_but_cat is included in cat_cols.  
    The total number of variables is equal to the sum of the three returned lists:
    cat_cols + num_cols + cat_but_car = total variable count
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]  

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]  

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]  

    cat_cols = cat_cols + num_but_cat  

    cat_cols = [col for col in cat_cols if col not in cat_but_car]  

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] 

    num_cols = [col for col in num_cols if col not in num_but_cat] 

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(cat_cols)
# ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
# 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
# 'PaymentMethod', 'Churn', 'SeniorCitizen']
print(num_cols)
# ['tenure', 'MonthlyCharges']
print(cat_but_car)
# ['customerID', 'TotalCharges']


#########################
# Step 2: Make necessary adjustments (e.g., variables with incorrect data types)
#########################

df.dtypes

"""
customerID           object
gender               object
SeniorCitizen         int64
Partner              object
Dependents           object
tenure                int64
PhoneService         object
MultipleLines        object
InternetService      object
OnlineSecurity       object
OnlineBackup         object
DeviceProtection     object
TechSupport          object
StreamingTV          object
StreamingMovies      object
Contract             object
PaperlessBilling     object
PaymentMethod        object
MonthlyCharges      float64
TotalCharges         object
Churn                object
"""




# customerID           object
# gender               object
# SeniorCitizen         int64 --> object 
# Partner              object
# Dependents           object
# tenure                int64
# PhoneService         object
# MultipleLines        object
# InternetService      object
# OnlineSecurity       object
# OnlineBackup         object
# DeviceProtection     object
# TechSupport          object
# StreamingTV          object
# StreamingMovies      object
# Contract             object
# PaperlessBilling     object
# PaymentMethod        object
# MonthlyCharges      float64
# TotalCharges         object --> float 
# Churn                object


df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)
# SeniorCitizen        object

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# TotalCharges        float64

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#########################
# Step 3: Observe the distribution of numerical and categorical variables in the dataset
#########################

# The classes of categorical variables and the proportions of these classes
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)
#        Churn      Ratio
# Churn
# No      5174  73.463013
# Yes     1869  26.536987



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)
plt.show()



#########################
# Step 4: Analyze the relationship between categorical variables and the target variable
#########################

df.groupby("Churn")["gender"].value_counts()
# Churn  gender
# No     Male      2625
#        Female    2549
# Yes    Female     939
#        Male       930

df.groupby("Churn")["Dependents"].value_counts()
# Churn  Dependents
# No     No            3390
#        Yes           1784
# Yes    No            1543
#        Yes            326

df.groupby("Churn")["Partner"].value_counts()
# Churn  Partner
# No     Yes        2733
#        No         2441
# Yes    No         1200
#        Yes         669

for col in cat_cols:
    print(df.groupby("Churn")[col].value_counts())

#########################
# ANALYSIS OF NUMERICAL VARIABLES WITH RESPECT TO THE TARGET VARIABLE
#########################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n")
    print("##########################################")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)
#        tenure
# Churn
# No     37.570
# Yes    17.979
# ##########################################
#        MonthlyCharges
# Churn
# No             61.265
# Yes            74.441
# ##########################################
#        TotalCharges
# Churn
# No         2555.344
# Yes        1531.796


#########################
# Step 5: Check for outliers 
#########################


for col in num_cols:
    print(col, check_outlier(df, col))
# tenure False
# MonthlyCharges False
# TotalCharges False


#########################
# Step 6: Check for missing values
#########################

df.isnull().sum()
# customerID           0
# gender               0
# SeniorCitizen        0
# Partner              0
# Dependents           0
# tenure               0
# PhoneService         0
# MultipleLines        0
# InternetService      0
# OnlineSecurity       0
# OnlineBackup         0
# DeviceProtection     0
# TechSupport          0
# StreamingTV          0
# StreamingMovies      0
# Contract             0
# PaperlessBilling     0
# PaymentMethod        0
# MonthlyCharges       0
# TotalCharges        11 --> 11 eksik gözlem
# Churn                0


###############################################################################
# Task 2 : Feature Engineering
###############################################################################

#########################
# Step 1: Perform necessary operations for missing and outlier observations
#########################

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
# TotalCharges        0
# Eksik değerler TotalCharges'ın mean değeri ile dolduruldu


#########################
# Step 2: Create new variables
#########################

df.head()

df["AnnualCharges"] = df["MonthlyCharges"] * 12 

labels = ["Low Charges", "Moderate Charges", "High Charges", "Very High Charges"]
df["ChargesCategory"] = pd.qcut(df["TotalCharges"], q=4, labels=labels)  

df["TenureYears"] = df["tenure"] / 12  

labels = ["New Customer", "Recent Customer", "Regular Customer", "Long-term Customer"]
df["CustomerType"] = pd.qcut(df["tenure"], q=4, labels=labels)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#########################
# Step 3: Perform encoding operations
#########################


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]
# nunique(): Varsayılan olarak eksik değerleri bir sınıf olarak saymaz, eksik değerler göz ardı edilir.
# len(df[col].unique()): Eksik değerleri de eşsiz bir sınıf olarak sayar, bu yüzden daha yüksek bir sayı dönebilir.

# "Churn" sütununu çıkar
binary_cols = [col for col in binary_cols if col not in "Churn"]


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe



for col in binary_cols:  # veri setindeki tüm 2 sınıfa sahip değişkenlere bu fonksiyonu uygula
    label_encoder(df, col)


# ONE-HOT ENCODING
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in "Churn"]


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    # dataframe = dataframe.astype(int)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)


df.head()
df.shape  # (7043, 40)

boolean_cols = df.select_dtypes(include=[bool]).columns
df[boolean_cols] = df[boolean_cols].astype(int)

df['Churn'] = df['Churn'].replace({'yes': 1, 'no': 0})
df["Churn"] = label_encoder(df,"Churn")

#########################
# Step 4: Perform standardization for numerical variables
#########################

num_cols  # ['tenure', 'MonthlyCharges', 'TotalCharges', 'AnnualCharges', 'TenureYears']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


###############################################################################
# Task 3: Modeling
###############################################################################

#########################
# Step 1: Build models using classification algorithms, evaluate their accuracy scores, and select the top 4 models
#########################

df = df.drop("customerID", axis=1)

# Modeling the relationship between the dependent (target) variable and the independent variables

y = df["Churn"]  # bağımlı değişken

X = df.drop(["Churn"], axis=1) # Drop the target (dependent) variable from the entire dataset = independent variables

log_model = LogisticRegression().fit(X, y) 

log_model.intercept_  # b: sabit değeri = array([-1.05233806])
log_model.coef_  # w: bağımsız değişkenlerin katsayısı/ağırlığı:
# array([[-0.01899633,  0.21609658,  0.01583725, -0.14866521, -0.492301  ,
#         -0.49042428,  0.3557046 , -0.04766854,  0.20098826, -0.04766854,
#         -0.492301  , -0.05399565,  0.33599301,  0.97513619, -0.13906303,
#         -0.13906303, -0.32739857, -0.13906303, -0.12174191, -0.13906303,
#          0.02150451, -0.13906303, -0.30463807, -0.13906303,  0.30228408,
#         -0.13906303,  0.32472967, -0.68166989, -1.47233463, -0.0910438 ,
#          0.27613185, -0.09041179, -0.42190094, -0.49726959, -0.50128862,
#         -0.23002033,  0.12749764,  0.28286889]])

y_pred = log_model.predict(X)


######################################################
# Model Evaluation
######################################################

###########################################
# 1. LOGISTIC REGRESSION:
############################################

# Numerical values of Accuracy, Precision, Recall, and F1-score from the confusion matrix and the corresponding heatmap
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))
#               precision    recall  f1-score   support
#           No       0.84      0.90      0.87      5174
#          Yes       0.67      0.53      0.59      1869 --> İlgilendiğimiz kısım: sınıfı 1 olanlara göre değerlendirme
#     accuracy                           0.81      7043
#    macro avg       0.76      0.72      0.73      7043
# weighted avg       0.80      0.81      0.80      7043


# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
# Probabilities of the target variable being class 1, calculated based on different classification thresholds
roc_auc_score(y, y_prob)  # 0.85 --> modelin pozitif (1) sınıfları negatif (0) sınıflardan ayırma yeteneği oldukça iyi


#######################
# 1.1. Model Validation: Holdout (Test) Approach
#######################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#           No       0.84      0.92      0.87      1030
#          Yes       0.69      0.51      0.59       379
#     accuracy                           0.81      1409
#    macro avg       0.76      0.71      0.73      1409
# weighted avg       0.80      0.81      0.80      1409

plot_roc_curve(log_model, X_test, y_test)  # HATA ALIYORUM
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)  # 0.83


#######################
# 1.2. Model Validation: 10-Fold Cross Validation
#######################

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model, X, y, cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
# {'fit_time': array([0.06749701, 0.0303998 , 0.03608608, 0.04254079, 0.02970529]),
#  'score_time': array([0.01817393, 0.01455736, 0.01404285, 0.01442909, 0.01426482]),
#  'test_accuracy': array([0.80411639, 0.80979418, 0.79488999, 0.8125    , 0.80610795]),
#  'test_precision': array([0.66554054, 0.67549669, 0.64026403, 0.68728522, 0.67474048]),
#  'test_recall': array([0.52673797, 0.54545455, 0.51871658, 0.53619303, 0.52139037]),
#  'test_f1': array([0.5880597 , 0.6035503 , 0.57311669, 0.60240964, 0.58823529]),
#  'test_roc_auc': array([0.86021855, 0.85844636, 0.83364721, 0.84165987, 0.84072808])}

cv_results['test_accuracy'].mean()  # 5 adet accuracy değerinin ortalamasını al
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.8054

cv_results['test_recall'].mean()
# Recall: 0.5296

cv_results['test_f1'].mean()
# F1-score: 0.5910

cv_results['test_roc_auc'].mean()
# AUC: 0.8469


###########################################
# 2. KNN MODEL:
###########################################

knn_model = KNeighborsClassifier().fit(X, y)

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)  

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
#               precision    recall  f1-score   support
#            0       0.88      0.91      0.89      5174
#            1       0.72      0.64      0.68      1869
#     accuracy                           0.84      7043
#    macro avg       0.80      0.78      0.79      7043
# weighted avg       0.84      0.84      0.84      7043

# AUC
roc_auc_score(y, y_prob)  # 0.9013


###########################
# 2.1. Model Validation: K-Fold Cross Validation
###########################

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.77
cv_results['test_f1'].mean()  # 0.559
cv_results['test_roc_auc'].mean()  # 0.784


#########################
# Step 2: Perform hyperparameter optimization for the selected models and rebuild the models using the optimized parameters
#########################

# hyperparameters 

knn_model = KNeighborsClassifier()
knn_model.get_params()
# {'algorithm': 'auto',
#  'leaf_size': 30,
#  'metric': 'minkowski',
#  'metric_params': None,
#  'n_jobs': None,
#  'n_neighbors': 5, --> komşuluk sayısı: 5
#  'p': 2,
#  'weights': 'uniform'}

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)
# Fitting 5 folds for each of 48 candidates, totalling 240 fits

knn_gs_best.best_params_  # {'n_neighbors': 42}


###########################
# Final Model
###########################

# Above, we tested which hyperparameter values work best for the KNN algorithm.  
# Now, we will rebuild the model using these values.

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,  X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()  # 0.797
cv_results['test_f1'].mean()  # 0.587
cv_results['test_roc_auc'].mean()  # 0.836

# Hiperparametre değerlerinin değiştirilmesinden sonra, 3 metrik için de değerler yükselmiş, modelin performansı artmış


###########################
# Prediction for A New Observation
###########################

random_user = X.sample(1, random_state=45)

log_model.predict(random_user)
