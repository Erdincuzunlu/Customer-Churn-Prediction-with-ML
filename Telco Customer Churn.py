##########################################################################
# Telco Churn Prediction
# -CASE STUDY I-
##########################################################################

# İŞ PROBLEMİ:
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri
# sağlayan hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını
# veya hizmete kaydolduğunu gösterir.

# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik)
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

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


# Veri setinde değişkenlerin eşik değerlerinin hesaplanması:
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Veri setinde değişkenlerde aykırı değer var mı yok mu?:
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Aykırı değere sahip değişkendeki aykırı değerlerin eşik değerleri ile değiştirilmesi:
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


###############################################################################
# Görev 1: Keşifçi Veri Analizi
###############################################################################

#########################
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
#########################

def load():  # Bu fonksiyon churn veri setini getirir
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

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal (bilgi taşımayan) değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir. (ör: Survived, Pclass)

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal (çok fazla sınıfı olan) değişkenler için sınıf eşik değeri (ör: Name, Ticket)

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]  # tipi kategorik olanların hepsi

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]  # tipi numerik görünen ama aslında kategorik olanlar (ör: Survived vs)

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]  # tipi kategorik görünen ama kardinal olanlar (ör: name)

    cat_cols = cat_cols + num_but_cat  # tüm kategorikler + kardinalleri de içerir.

    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # kardinalleri de çıkarttık sadece kategorik kaldı

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]  # tipi numerik olanların hepsi

    num_cols = [col for col in num_cols if col not in num_but_cat]  # tüm num. olanlardan; num. görünüp cat.'leri çıkart

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
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
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
# SeniorCitizen         int64 --> object olmalı
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
# TotalCharges         object --> float olmalı
# Churn                object


df["SeniorCitizen"] = df["SeniorCitizen"].astype(object)
# SeniorCitizen        object

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# TotalCharges        float64

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#########################
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
#########################

# Kategorik değişkenlerin sınıflarını ve bu sınıfların oranları:
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


# Numerik değişkenlerin yüzdelik değerleri
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

##########BERNA######
def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"TARGET FREQUENCY ": dataframe.groupby(cat_col)[target].count(), "RATIO ": 100 * dataframe.groupby(cat_col)[target].count() /
                                                                                                   len(dataframe)}))

for col in cat_cols: target_summary_with_cat(df, "Churn", col)
###############



# count   7043.000
# mean      32.371
# std       24.559
# min        0.000
# 5%         1.000
# 10%        2.000
# 20%        6.000
# 30%       12.000
# 40%       20.000
# 50%       29.000
# 60%       40.000
# 70%       50.000
# 80%       60.000
# 90%       69.000
# 95%       72.000
# 99%       72.000
# max       72.000
# Name: tenure, dtype: float64
# ##########################################
# count   7043.000
# mean      64.762
# std       30.090
# min       18.250
# 5%        19.650
# 10%       20.050
# 20%       25.050
# 30%       45.850
# 40%       58.830
# 50%       70.350
# 60%       79.100
# 70%       85.500
# 80%       94.250
# 90%      102.600
# 95%      107.400
# 99%      114.729
# max      118.750
# Name: MonthlyCharges, dtype: float64
# ##########################################
# count   7032.000
# mean    2283.300
# std     2266.771
# min       18.800
# 5%        49.605
# 10%       84.600
# 20%      267.070
# 30%      551.995
# 40%      944.170
# 50%     1397.475
# 60%     2048.950
# 70%     3141.130
# 80%     4475.410
# 90%     5976.640
# 95%     6923.590
# 99%     8039.883
# max     8684.800
# Name: TotalCharges, dtype: float64


#########################
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
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
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
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
# Adım 5: Aykırı gözlem var mı inceleyiniz.
#########################

# Veri setinde değişkenlerde aykırı değer var mı yok mu?

for col in num_cols:
    print(col, check_outlier(df, col))
# tenure False
# MonthlyCharges False
# TotalCharges False

# Veri setinde hiçbir değişkende aykırı değer yok

#########################
# Adım 6: Eksik gözlem var mı inceleyiniz.
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
# Görev 2 : Feature Engineering
###############################################################################

#########################
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
#########################

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
# TotalCharges        0
# Eksik değerler TotalCharges'ın mean değeri ile dolduruldu

#######AKAY#######

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce') #Sayısal olmayan (string) değerleri NaN olarak işaretleyelim
df["TotalCharges"].isnull().sum()   #Out[33]: 11  adet NaN değer var
df = df.dropna(subset=["TotalCharges"])
####################

#########################
# Adım 2: Yeni değişkenler oluşturunuz.
#########################

df.head()

df["AnnualCharges"] = df["MonthlyCharges"] * 12  # Müşteriden yıllık olarak tahsil edilen tutar

labels = ["Low Charges", "Moderate Charges", "High Charges", "Very High Charges"]
df["ChargesCategory"] = pd.qcut(df["TotalCharges"], q=4, labels=labels)  # TotalCharges değişkeni değerlerini 4'e böler

df["TenureYears"] = df["tenure"] / 12  # Müşterinin şirkette kaldığı yıl sayısı

labels = ["New Customer", "Recent Customer", "Regular Customer", "Long-term Customer"]
df["CustomerType"] = pd.qcut(df["tenure"], q=4, labels=labels)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
########BERNA############
df['NEW_Contract_Length'] = [0 if x == 'Month-to-month' else 1 if x == 'One year' else 2 for x in df['Contract']]



########AKAY#####

df["Contract"].unique()  #Out[90]: array(['Month-to-month', 'One year', 'Two year'], dtype=object)
odeme_tipi = ["Month-to-month", "One year", "Two year"]
df["Contract"] = pd.Categorical(df["Contract"], categories=odeme_tipi, ordered=True) ## Categorical veri tipi ile sıralama
df["Contract_Encoded"] = df["Contract"].cat.codes ##
########################
#########################
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
#########################

# 2 sınıfa sahip olan tipi kategorik olan değişkenleri BINARY ENCODER'dan geçireceğiz:

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
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
#########################

num_cols  # ['tenure', 'MonthlyCharges', 'TotalCharges', 'AnnualCharges', 'TenureYears']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


###############################################################################
# Görev 3 : Modelleme
###############################################################################

#########################
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
#########################

df = df.drop("customerID", axis=1)

# Bağımlı (hedef) değişken ile bağımsız değişkenler arasındaki ilişkinin modellenmesi:

y = df["Churn"]  # bağımlı değişken

X = df.drop(["Churn"], axis=1)  # Tüm veri setinden bağımlı değişkeni düşür = bağımsız değişkenler

log_model = LogisticRegression().fit(X, y)  # Logistik regresyon modelin kurulup X ve y'ye fit edilmesi

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
# 1. YÖNTEM: LOGISTIC REGRESSION:
############################################

# Karmaşıklık matrixte bulunan Accuracy, Precision, Recall, F1-score değerlerin numerik karşılıkları ve heatmapi:
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
# Olası farklı sınıflandırma tresholdlarına göre hesaplanan bağımlı değişkenin 1 sınıfının gerçekleşme olasılıkları
roc_auc_score(y, y_prob)  # 0.85 --> modelin pozitif (1) sınıfları negatif (0) sınıflardan ayırma yeteneği oldukça iyi


#######################
# 1.1. Model Validation (Doğrulama): Holdout (Sınama) Yaklaşımı
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
# 2. YÖNTEM: KNN MODEL:
###########################################

knn_model = KNeighborsClassifier().fit(X, y)

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)  # bütün gözlem birimleri için tahmin yap ve bunları y_pred'de sakla

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
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile
# modeli tekrar kurunuz.
#########################

# hyperparameters (veri setinden bulunmayan kullanıcı tarafından dışardan belirlenen parametreler)

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

# Yukarıda KNN algoritmasının en iyi hangi hiperparametre değeri ile çalışacağını test ettik. Şimdi bu değerleri
# kullanarak modeli tekrar kuracağız.

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
