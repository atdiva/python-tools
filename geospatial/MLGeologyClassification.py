import numpy as np 
import matplotlib.pyplot as plt
import geopandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import sklearn 
import pandas as pd
import seaborn as sns


# Given a set of features:
# derived_era, area_m2, belt (categroical)
# develop a ML classifer to predict wether
# a sample is one of the rock classes: 
# [sedimentary, metamorphic]

##############################################################################################################################
#Data prepareation 
df = geopandas.read_file("../Geo/BedrockP.gpkg")

##############################################################################################################################
#Map categorical era->numerical
era_to_num = {"default": np.nan, 
              "Cenozoic": 66*(10**6), 
              "Mesozoic": 186*(10**6),
              "Paleozoic": 400*(10**6) } 
df['derived_era'] = [ np.random.normal(era_to_num[era], 10**7) if era in era_to_num else era_to_num['default'] for era in df['era'] ] 
#remove rows that are nan 
df = df[ ~np.isnan(df['derived_era']) ]

#log large numbers
df['derived_era'] = np.log10(df['derived_era'])
df['area_m2'] = np.log10(df['area_m2'])

#scale by standardizing
df['derived_era'] = (df['derived_era'] - np.mean(df['derived_era']))/np.std(df['derived_era'])
df['area_m2'] = (df['area_m2'] - np.mean(df['area_m2']))/np.std(df['area_m2'])

#limit data set
df = df[ ['derived_era', 'area_m2', 'belt', 'rock_class'] ]
#reduce dataframe to rock types of interest
df = df[ (df['rock_class'].str.contains("sedimentary")) | (df['rock_class'].str.contains("metamorphic")) ]

#encode period 
df = pd.get_dummies(df, columns=['belt'], drop_first=True)
#rename one-hot encode 
df = df.rename(columns = {"belt_Intermontane": "belt" } )

#rock_class map: sedimentary = 0, metamorphic = 1, 
df['rock_class_boolean'] = [ 1 if "metamorphic" in rock_class else 0 for rock_class in df['rock_class'] ]

sed_index = (df['rock_class_boolean'] == 0)
met_index = (df['rock_class_boolean'] == 1)
plt.figure()
plt.scatter(df.loc[ sed_index, ['area_m2']], df.loc[ sed_index, 'derived_era'], label="sedimentary")
plt.scatter(df.loc[ met_index, ['area_m2']], df.loc[ met_index, 'derived_era'], label="metamorphic")
plt.legend()
plt.xlabel("log(area (m2))")
plt.ylabel("derived era log(yr)")
plt.grid()
plt.figure()
sns.countplot(x='belt', hue='rock_class', data=df)
plt.grid()

##############################################################################################################################
model_type = "logistic"
features = ['derived_era', 'area_m2', 'belt']
target = 'rock_class_boolean'

df_features = df[features]
print("corr of features:")
print(df_features.corr())

#train/test/split
train_df, test_df = train_test_split(
    df,
    test_size=0.6,
    stratify= df[target],
    random_state=42
)

#Train 
X_train = train_df[features]
y_train = train_df[target]

if model_type == "logistic":
    model = LogisticRegression().fit(X_train, y_train)
    print("Logistic Coeff " + str(model.coef_) )
else:
    #KNN
    model = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)

#Evaluate (0.5 is as good as a random classifier, anything greater area under curve is greater than 0.5)
auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

#Training Metrics
print("AUC: " + str(auc) )

#Testing Metrics 
X_test = test_df[features]
y_test = test_df[target]
#predict
predictions = model.predict(X_test)
conf_mat = confusion_matrix(y_test, predictions)
print("Confusion Matrix Details")
print(conf_mat)

#recall, precision 
recall = sklearn.metrics.recall_score(y_test, predictions)
precision = sklearn.metrics.precision_score(y_test, predictions)
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
f1 = sklearn.metrics.f1_score(y_test, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)

#preidct probability for a new sample 
data = {"derived_era":2.0, "area_m2":0.0, "belt":False}
df_new_sample = pd.DataFrame(data, index=[0])
print( model.predict_proba( np.array(df_new_sample).reshape(1, -1) ) )

