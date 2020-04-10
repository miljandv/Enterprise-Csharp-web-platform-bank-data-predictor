import pandas as pd
import numpy as np
import sklearn as sl
from matplotlib import pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

#Procitaj podatke iz csv fajla
customers_df = pd.read_csv(r"C:\Users\milja\source\repos\saga\saga\Scripts\customers.csv", sep=';')
customers = customers_df.values
transactions_cards=pd.read_csv(r"C:\Users\milja\source\repos\saga\saga\Scripts\transactions_cards.csv",sep=';')
transactions_domestic=pd.read_csv(r"C:\Users\milja\source\repos\saga\saga\Scripts\transactions_domestic.csv",sep=';')
products = pd.read_csv(r"C:\Users\milja\source\repos\saga\saga\Scripts\products.csv", sep=';')

#kreiranje dataset-a za treniranje modela
dataset = np.zeros((len(customers), 1))
dataset = customers_df['CustomerId'].values

#odredjivanje broja godina korisnika
now = datetime.datetime.now()
cur_year = now.year
dataset = np.vstack((dataset, cur_year - customers_df['BirthDate'].str[:4].astype(int)))
dataset = dataset.T

#kodiranje odlike LivingConditions
living_cond = customers_df['LivingConditions'].astype("category").cat.codes
living_cond = living_cond.values.reshape(-1,1)
dataset = np.hstack((dataset, living_cond))

#Kodiranje odlike EmploymentStatus
employment = customers_df['EmploymentStatus'].astype("category").cat.codes
employment = employment.values.reshape(-1,1)
dataset = np.hstack((dataset, employment))

#Kodiranje odlike MaritalStatus
marital = customers_df['MaritalStatus'].astype("category").cat.codes
marital = marital.values.reshape(-1,1)
dataset = np.hstack((dataset, marital))

#Odredjivanje srednje vrednosti Gross income odlike
gi_avg = np.mean(customers_df.values[:,7:10], axis=1)
gi_avg = gi_avg.astype(dtype=np.float64)
gi_avg = gi_avg.reshape(-1,1)
dataset = np.hstack((dataset, gi_avg))
dataset_df = pd.DataFrame(dataset, columns=['CustomerId', 'BirthDate', 'LivingConditions', 'EmploymentStatus', 'MaritalStatus', 'AVG(GI)'])

krediti=products[products.ProductType.str.contains('Loan')].copy()

trans = pd.merge(transactions_cards, transactions_domestic, on='CustomerId', how='left')
trans = trans.groupby('CustomerId').sum().reset_index()
trans['total'] = (trans['AmountEur_x'] + trans['AmountEur_y'])/3
trans = trans[['CustomerId', 'total']]
dataset_df = pd.merge(dataset_df, trans, on='CustomerId', how='left')
dataset_df['total'] = dataset_df['total'].fillna(0)

dataset_df_stam = dataset_df.copy()
dataset_df_got = dataset_df.copy()

#Odredjivanje labela za stambene kredite
#Da li ima kredit trenutno
krediti=products[products.ProductType.str.contains('Loan')].copy()
stambeni_id = krediti[krediti.LoanType.str.contains('Stambeni')].copy()
date_time_str = max(products[products['LoanType'].isna() == False]['DateOpened'])
date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
dateBorder = date_time_obj - datetime.timedelta(6*365/12)
dateBorder = dateBorder.strftime('%Y-%m-%d')
print('DateBorder: ', dateBorder) 
stambeni_id = stambeni_id[stambeni_id.DateOpened > dateBorder]
stambeni_id = stambeni_id.groupby(['CustomerId']).size().reset_index(name='brStambenih')
dataset_df_stam = pd.merge(dataset_df_stam, stambeni_id, on='CustomerId', how='left')
dataset_df_stam['brStambenih'] = dataset_df_stam['brStambenih'].fillna(0)
dataset_df_stam['brStambenih'] = dataset_df_stam['brStambenih'].astype(int)
dataset_df_stam['brStambenih'] = dataset_df_stam['brStambenih'].replace(2, 1)
dataset_stam = dataset_df_stam.values

#Odredjivanje labela za gotovinske kredite
gotovinski_id = krediti[krediti.LoanType.str.contains('Gotovinski')].copy()
gotovinski_id = gotovinski_id[gotovinski_id.DateOpened > dateBorder]
gotovinski_id = gotovinski_id.groupby(['CustomerId']).size().reset_index(name='brGotovinskih')
dataset_df_got = pd.merge(dataset_df_got, gotovinski_id, on='CustomerId', how='left')
dataset_df_got['brGotovinskih'] = dataset_df_got['brGotovinskih'].fillna(0)
dataset_df_got['brGotovinskih'] = dataset_df_got['brGotovinskih'].astype(int)
dataset_df_got['brGotovinskih'].values[dataset_df_got['brGotovinskih'].values > 1] = 1 #Sve vrednosti vece od 1 svedene na 1
dataset_got = dataset_df_got.values

##izdvajanje prediktora
x = dataset_stam[:,1:7]
scaler = StandardScaler()
#izdvajanje labela
y_stam = dataset_stam[:,7]
y_got = dataset_got[:,7]
#podela na trening i test skup
X_train = x[:33200, :]
X_train = scaler.fit_transform(X_train)
X_valid = x[33200:42700, :]
X_valid = scaler.transform(X_valid)
X_test = x[42700:, :]
X_test = scaler.transform(X_test)
y_train_stam = y_stam[:33200]
y_valid_stam = y_stam[33200:42700]
y_test_stam = y_stam[42700:]
y_train_got = y_got[:33200]
y_valid_got = y_got[33200:42700]
y_test_got = y_got[42700:]

##treniranje modela za stambene kredite
model_stam = RandomForestClassifier(n_estimators=1000, bootstrap=True, class_weight='balanced',  
                               max_features=3, criterion='entropy', max_depth=5)
model_stam.fit(X_train, y_train_stam)
#testiranje modela za stambene kredite
y_pred_stam = model_stam.predict(X_valid)
threshold_stam = 0.5
predicted_proba = model_stam.predict_proba(X_valid)
predicted = (predicted_proba [:,1] >= threshold_stam).astype('int')
print('F1 skor na testirajućem skupu: ', 
      precision_recall_fscore_support(y_valid_stam, predicted, average=None, labels=[1]))
fpr, tpr, thresholds = roc_curve(y_valid_stam, predicted)
plt.title("ROC")
plt.xlabel("LP")
plt.ylabel("TP")
plt.plot(fpr, tpr, color="navy")
plt.show()
print('AUC score: ', sl.metrics.auc(fpr, tpr))
plt.barh(range(x.shape[1]), model_stam.feature_importances_)
plt.show()

##treniranje modela za gotovinske kredite
model_got = RandomForestClassifier(n_estimators=1000, bootstrap=True, class_weight='balanced',  
                               max_features=3, criterion='entropy', max_depth=5)
model_got.fit(X_train, y_train_got)
#testiranje modela za gotovinske kredite
y_pred_got = model_got.predict(X_valid)
threshold = 0.5
predicted_proba = model_got.predict_proba(X_valid)
predicted = (predicted_proba [:,1] >= threshold).astype('int')
print('F1 skor na testirajućem skupu: ', 
      precision_recall_fscore_support(y_valid_got, predicted, average=None, labels=[1]))
fpr, tpr, thresholds = roc_curve(y_valid_got, predicted)
plt.title("ROC")
plt.xlabel("LP")
plt.ylabel("TP")
plt.plot(fpr, tpr, color="navy")
plt.show()
print('AUC score: ', sl.metrics.auc(fpr, tpr))
plt.barh(range(x.shape[1]), model_got.feature_importances_)
plt.show()


transakcije = pd.merge(transactions_cards, transactions_domestic, on='CustomerId', how='left')
transakcije = transakcije.groupby(['CustomerId','MerchantTypeName']).sum().reset_index()
transakcije['total'] = transakcije['AmountEur_x'] + transakcije['AmountEur_y']
print(transakcije)

def potrosnja(customerId):
    zaKupca = transakcije[transakcije['CustomerId']==customerId].sort_values(by='total',ascending=False)
    plt.gcf().subplots_adjust(left=0.6)
    plt.barh(zaKupca['MerchantTypeName'], zaKupca['total'], align='center', alpha=0.5)
    plt.savefig(r"C:\Users\milja\source\repos\saga\saga\wwwroot\images\graph.jpg")
 
print('start loop')
while True:
    numbers = []
    while True:
        numbers = []
        with open(r"C:\Users\milja\source\repos\saga\saga\Output\newdata.txt") as fp:
            for line in fp:
                numbers.extend(
                    [float(item)
                    for item in line.split()
                    ])
        if len(numbers)==2:
            break
    print(numbers)
    open(r"C:\Users\milja\source\repos\saga\saga\Output\newdata.txt", 'w').close()
    if numbers[0]==-1:
        print('-1')
        #f = open(r"C:\Users\milja\source\repos\saga\saga\Output\result.txt", "w")
        #f.write("" + str(numbers[1]*10) + "\n")
        #f.close()
        potrosnja(numbers[1])
    elif numbers[0]==-2:
        print('-2')
        f = open(r"C:\Users\milja\source\repos\saga\saga\Output\result.txt", "w")
        f.write(str(numbers[1]*10) + " "+str(numbers[1]*10) + " "+str(numbers[1]*10) + " "+
        str(numbers[1]*10) + " " +str(numbers[1]*10) + " " +str(numbers[1]*10) + "\n")
        f.close()
  #  y_pred = model.predict(X_test)
    



