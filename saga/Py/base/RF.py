import pandas as pd
import numpy as np
import sklearn as sl
from matplotlib import pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve


def standardize(x):
    #standardizacija prediktora
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_stand = np.subtract(x, x_mean)
    x_stand = np.divide(x_stand, x_std)
    return x_stand

#Procitaj podatke iz csv fajla
customers_df = pd.read_csv('customers.csv', sep=';')
#promesaj podatke
customers_df = customers_df.sample(frac=1).reset_index(drop=True)
customers = customers_df.values

products = pd.read_csv('products.csv', sep=';')
products = products.sample(frac=1).reset_index(drop=True)

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
#Trenutni broj kredita korisnika
krediti=products[products.ProductType.str.contains('Loan')].copy()
#otvoreni_krediti=krediti[krediti['DateClosed'].isnull()]
krediti['DateClosed'] = krediti['DateClosed'].fillna('1')
otvoreni_krediti = krediti[~(krediti.DateClosed != '1')].copy()
otvoreni_krediti =  otvoreni_krediti[['CustomerId', 'DateClosed']]
otvoreni_krediti['DateClosed'] = otvoreni_krediti['DateClosed'].astype(int)
dataset_df = pd.merge(dataset_df, otvoreni_krediti, on='CustomerId', how='left')
dataset_df['DateClosed'] = dataset_df['DateClosed'].fillna(0)

#krediti_po_id=krediti_bez_ss[krediti_bez_ss.ProductType.str.contains('Loan')]
#dataset_df = pd.merge(dataset_df, krediti_po_id, on='CustomerId', how='left')
#dataset_df['count'] = dataset_df['count'].fillna(0)
#dataset_df['count'] = dataset_df['count'].astype(int)

stambeni_id = krediti[krediti.LoanType.str.contains('Stambeni')].copy()
stambeni_id = stambeni_id.groupby(['CustomerId']).size().reset_index(name='brStambenih')
dataset_df = pd.merge(dataset_df, stambeni_id, on='CustomerId', how='left')
dataset_df['brStambenih'] = dataset_df['brStambenih'].fillna(0)
dataset_df['brStambenih'] = dataset_df['brStambenih'].astype(int)
dataset_df['brStambenih'] = dataset_df['brStambenih'].replace(2, 1)
dataset = dataset_df.values

#izdvajanje prediktora
x = dataset[:,1:7]
x = standardize(x)
#izdvajanje labela
y = dataset[:,7]
#podela na trening i test skup
X_train = x[:38000, :]
X_test = x[38000:, :]
y_train = y[:38000]
y_test = y[38000:]


#treniranje modela
#model = LogisticRegression(class_weight='balanced', random_state=0, penalty='l2', C=0.1)
model = RandomForestClassifier(n_estimators=5000, bootstrap=True, class_weight='balanced',  
                               max_features=3, criterion='entropy', max_depth=5)
model.fit(X_train, y_train)
# Performanse na obučavajućem skupu
y_pred_train = model.predict(X_train)
print('F1 skor na trening skupu: ',
      precision_recall_fscore_support(y_train, y_pred_train, average=None, labels=[1]))

# Performanse na testirajućem skupu
y_pred = model.predict(X_test)
print('F1 skor na testirajućem skupu: ', 
      precision_recall_fscore_support(y_test, y_pred, average=None, labels=[1]))

procena = model.predict_proba(X_test)
#print(procena)

procena_ind = np.array(np.where(procena[:,1] > 0.8)).ravel()
print(procena_ind)
print('Broj onih sa verovatnocom preko 0.8: ', len(procena_ind))
true_ind = np.array(np.where(y_test == 1)).ravel()
print('Indeksi tacnih jedinica: ')
print(true_ind)

fpr, tpr, thresholds = roc_curve(y_test, y_pred)







