#Importing the neccessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import preprocessing
import datetime
#Importing the neccessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

#Importing our dataset
df_cab=pd.read_csv('Datasets/cab_rides.csv')
#Creating an index label for the dataset
df_cab.reset_index(drop=True, inplace=True)
df_cab.info()

df_cab.shape
df_cab = df_cab.iloc[0:30000, :]
df_cab.shape

df_cab.head(2)

df_cab.describe()
#Data Cleaning and Preprocessing
df_cab.isnull().sum()
df_cab['cab_type'].value_counts()
df_cab['destination'].value_counts()
df_cab['source'].value_counts()
df_cab['surge_multiplier'].value_counts()
df_cab['name'].value_counts()

import datetime
#convert 13digit time stamp to datetime format
df_cab['date_time']= pd.to_datetime(df_cab['time_stamp']/1000, unit='s')
df_cab['date']= df_cab['date_time'].dt.date
df_cab['day'] = df_cab.date_time.dt.dayofweek
df_cab['hour'] = df_cab.date_time.dt.hour
#extract hours only
df_cab['fare_per_mile']= round(df_cab.price/df_cab.distance,2)

#After conversion to a new column time we can delete the exisitng column
del df_cab['time_stamp']
#Replacing null values.
df_cab['fare_per_mile']=df_cab['fare_per_mile'].astype(float)
df_cab['fare_per_mile'].fillna('2.8',inplace=True)
df_cab['price'] = df_cab['fare_per_mile']*df_cab['distance']
#After replacement checking if the change is reflected in the dataset.
df_cab.price.isnull().sum()

#Visualisations
#Mentioning our graph sizes
from pylab import rcParams
rcParams['figure.figsize'] = 16,10
sns.heatmap(df_cab.describe()[1:].transpose(),annot=True,linecolor='w',linewidth=2,cmap=sns.color_palette('Paired'))
plt.title("DATA SUMMARY")

#Heatmap from searborn library
df_corr = df_cab.corr()
sns.heatmap(df_corr, cmap =sns.color_palette("Set3"),annot = True)
plt.title("CORRELATION BETWEEN VARIABLES")

#Barplot  to display Uber Vs Lyft 

plt.figure(figsize=(10,8))
flatui = [ "#3498db", "#2ecc71"]
x=['Uber','Lyft']
y = [df_cab.cab_type[(df_cab.cab_type)=='Uber'].count(),df_cab.cab_type[(df_cab.cab_type)=='Lyft'].count()]
vis1= sns.barplot(x,y,palette=flatui)

#Data Preparation for PReprocessing
#Import the new dataset and view the attributes
df_weather=pd.read_csv('Datasets/weather.csv')
df_weather.head(2)

df_weather['rain'].fillna(0, inplace = True)

#Splitting the time_stamp attribute in weather to two attributes which are time and data
df_weather['date_time'] = pd.to_datetime(df_weather['time_stamp'], unit='s')
del df_weather['time_stamp']

#df_weather['date_time']

#merge the datasets to refelect same time for a location
df_cab['merge_date'] = df_cab.source.astype(str) +" - "+ df_cab.date_time.dt.date.astype("str") +" - "+ df_cab.date_time.dt.hour.astype("str")
df_weather['merge_date'] = df_weather.location.astype(str) +" - "+ df_weather.date_time.dt.date.astype("str") +" - "+ df_weather.date_time.dt.hour.astype("str")
print(df_cab.head(2))
print(df_weather.head(2))

df_weather = df_weather.groupby(['merge_date']).mean()
df_weather.reset_index(inplace=True)
df_weather.head()

#Merging based on the date and location.
df_merged = pd.merge(df_cab, df_weather, on='merge_date')
print(df_merged.shape)

#Visualisation of Merged Dataset
sns.heatmap(df_merged.describe()[1:].transpose(),annot=True,linecolor='w',linewidth=2,cmap=sns.color_palette('Paired'))
plt.title("DATA SUMMARY")

df_corr = df_merged.corr()
sns.heatmap(df_corr, cmap =sns.color_palette("Set3"),annot = True)
plt.title("CORRELATION BETWEEN VARIABLES")

#Catplot
sns.catplot(x="name", y="price", data=df_merged,kind="boxen", height=8, aspect=2);
sns.boxplot(data=df_merged, x='source',y='price',palette='Blues')


#Dropping certain columns as they are not neccessary for predictions
df_merged = df_merged.drop(['date_time','id','product_id'], axis=1)
f_merged = df_merged.drop(['fare_per_mile','surge_multiplier'],axis=1)
df_merged = df_merged.loc[:, df_merged.columns !='merge_date']

#Splitting the dataset
OverallData = df_merged.drop(['cab_type'],axis=1)
uber = df_merged[df_merged['cab_type']=='Uber']
uber.reset_index(inplace=True)
uber.drop('index', axis=1, inplace=True)
lyft = df_merged[df_merged['cab_type']=='Lyft']
lyft.reset_index(inplace=True)
lyft.drop('index', axis=1, inplace=True)

#Removing the cab type from uber and lyft
uber.drop('cab_type', axis=1, inplace=True)
lyft.drop('cab_type', axis=1, inplace=True)

#Overview of all the columns present
print(OverallData.columns)
print(uber.columns)
print(lyft.columns)

#Transformations in Data Preprocessing
Xd = OverallData.loc[:, OverallData.columns != 'price']
yd = OverallData['price']
Xd_train,Xd_test,yd_train,yd_test = train_test_split(Xd,yd,test_size = 0.33, random_state=42)

Xu = uber.loc[:, uber.columns != 'price']
yu = uber['price']
Xu_train,Xu_test,yu_train,yu_test = train_test_split(Xu,yu,test_size = 0.33, random_state=42)

Xl = lyft.loc[:, lyft.columns != 'price']
yl = lyft['price']
Xl_train,Xl_test,yl_train,yl_test = train_test_split(Xl,yl,test_size = 0.33, random_state=42)

#Modelling
numerical_features = Xu.dtypes == 'float'
categorical_features = ~numerical_features
preprocess = make_column_transformer(
(numerical_features, StandardScaler()),
(categorical_features, OneHotEncoder()))
rt = RandomForestRegressor(n_estimators=200,random_state = 42)

#Modelling for uber
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import metrics

pipe = make_pipeline(preprocess, rt)
pipe.fit(Xu_train, yu_train)
yu_pred = pipe.predict(Xu_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(yu_test, yu_pred))
print('Mean Squared Error:', metrics.mean_squared_error(yu_test, yu_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yu_test, yu_pred)))
print('Mean Absolute Percentage Error:', np.mean(np.abs((yu_test - yu_pred) / yu_test)) * 100)

#Modelling for lyft
pipe = make_pipeline(preprocess, rt)
pipe.fit(Xl_train, yl_train)
yl_pred = pipe.predict(Xl_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(yl_test, yl_pred))
print('Mean Squared Error:', metrics.mean_squared_error(yu_test, yu_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yl_test, yl_pred)))
print('Mean Absolute Percentage Error:', np.mean(np.abs((yl_test - yl_pred) / yl_test)) * 100)

#Modelling for Complete dataset
pipe = make_pipeline(preprocess, rt)
pipe.fit(Xd_train, yd_train)
yd_pred = pipe.predict(Xd_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(yd_test, yd_pred))
print('Mean Squared Error:', metrics.mean_squared_error(yd_test, yd_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yd_test, yd_pred)))
print('Mean Absolute Percentage Error:', np.mean(np.abs((yd_test - yd_pred) / yd_test)) * 100)


#Five-cross validation
from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = pipe, X = Xu, y = yu, cv = 5)
print("Mean Accuracy :", cvs.mean()*100)
print("Mean Standard Deviation :", cvs.std())

cvs = cross_val_score(estimator = pipe, X = Xl, y = yl, cv = 5)
print("Mean Accuracy :", cvs.mean()*100)
print("Mean Standard Deviation :", cvs.std())

cvs = cross_val_score(estimator = pipe, X = Xd, y = yd, cv = 5)

#Applying Apriori
rules = apriori(records, min_support=0.01, min_confidence=0.6, min_lift=4, min_length=2)

for item in rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("******************************")

print("Mean Accuracy :", cvs.mean()*100)
print("Mean Standard Deviation :", cvs.std())
