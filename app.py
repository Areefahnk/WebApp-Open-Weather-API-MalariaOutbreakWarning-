from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model/model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    float_features = [(x) for x in request.form.values()]
    final = [np.array(float_features)]
    #print(final)
    c= final[0]
    import requests
    from datetime import datetime
    import pytz
    city = c[2]
    print(city)
    date_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")
    print(date_time)
    api_key1 = 'ad62ecebb7931902c9fdbfefb78f3277'
    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric'.format(city, api_key1)
    res = requests.get(url)
    data = res.json()

    #print(f"Place : {data['name']}")
    latitude = data['coord']['lat']
    longitude = data['coord']['lon']
    print('latitude :', latitude)
    print('longitude :', longitude)
    # getting the main dict block
    main = data['main']
    wind = data['wind']
    # getting temperature
    temperature = main['temp']
    # getting the humidity
    humidity = main['humidity']
    tempmin = main['temp_min']
    tempmax = main['temp_max']
    # getting the pressure
    windspeed = wind['speed']
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

    inputt = []
    inputt.append(tempmax)
    inputt.append(tempmin)
    inputt.append(humidity)
    final = [np.array(inputt)]
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)

    if output>=str(0.5):
        return render_template('index.html',pred='There are more chances of Malaria Outbreak\nProbability of Malaria Breeds occuring is {}\n '.format(output),
                               inp="The Prediction is based on: Max Temp: %.2f,Min Temp: %.2f,Humidity: %.1f Units(Celcius,Percent)"%(inputt[0],inputt[1],inputt[2]))
    else:
        return render_template('index.html',pred='Less chance of Malaria Outbreak\n Probability of Malaria Breeds occuring is {}\n '.format(output),
                               inp="The Prediction is based on: Max Temp: %.2f,Min Temp: %.2f,Humidity: %.1f Units(Celcius,Percent)"%(inputt[0],inputt[1],inputt[2]))


if __name__ == '__main__':
    app.run(debug=True)
