import requests
from datetime import datetime
import pytz
city='warangal'

date_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")
print(date_time)
api_key1 = 'ad62ecebb7931902c9fdbfefb78f3277'
url= 'http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric'.format(city,api_key1)
res= requests.get(url)
data=res.json()

print(f"Place : {data['name']}")
latitude = data['coord']['lat']
longitude = data['coord']['lon']
print('latitude :',latitude)
print('longitude :',longitude)
   # getting the main dict block
main = data['main']
wind = data['wind']
  # getting temperature
temperature = main['temp']
  # getting the humidity
humidity = main['humidity']
tempmin=main['temp_min']
tempmax=main['temp_max']
  # getting the pressure
windspeed=wind['speed']
pressure = main['pressure']
  # weather report
report = data['weather']
print(f"Temperature : {temperature}°C")
print(f"Temperature Min : {tempmin}")
print(f"Temperature Max : {tempmax}")
print(f"Humidity : {humidity}")
print(f"Pressure : {pressure}")
print(f"Wind Speed : {windspeed}")
print(f"Weather Report : {report[0]['description']}")

#######################################
import numpy as np
import pandas as pd
import math
import random as rn
#import matplotlib.pyplot as plt
import pickle


data = pd.read_csv('outbreak_detect.csv')


# importing the preprocessing module from scikit-learn

from sklearn import preprocessing
LE= preprocessing.LabelEncoder()

# Fitting it to our dataset

data.Outbreak = LE.fit_transform(data.Outbreak)
data.head()

data = data.drop('Positive',axis=1)
data = data.drop('pf',axis=1)
data = data.drop('Rainfall',axis=1)

#Training
x = data.iloc[:,0:3]
y = data.iloc[:,3:]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(x,y)




inputt=[]
inputt.append(tempmax)
inputt.append(tempmin)
inputt.append(humidity)
final = [np.array(inputt)]
print(final)
prediction=clf.predict_proba(final)
output = '{0:.{1}f}'.format(prediction[0][1], 2)
print(output)
if output>=str(0.5):
  print(1)
else:
  print(0)


pickle.dump(clf,open('model/model.pkl','wb'))
model=pickle.load(open('model/model.pkl','rb'))
print("SUcess loaded")


