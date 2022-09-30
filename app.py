# importing the required libraries
from flask import Flask, request, jsonify, render_template
import dill
app = Flask(__name__)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
le = LabelEncoder()
oneH = OneHotEncoder()
import pickle

# home page - routing to the home page is done
@app.route('/home')
@app.route('/')
def home():
    # renders the home page template
    return render_template('index.html')


# diabetets page - routing to diabetes prediction page
@app.route('/d')
def d():
    # renders the diabetes prediction page template
    return render_template('index.html')


# portion for diabetes prediction
@app.route('/di', methods=['POST'])
def diabetespredict():
    with open('Bus_Decision_Tree_Classifier.joblib', 'rb') as io:
        tree_clf = dill.load(io)

    Pkl_Filename = "bus_encoder.pkl"
    with open(Pkl_Filename, 'rb') as file:
        busEncoder = pickle.load(file)

    int_features = [x for x in request.form.values()]
    print(int_features)

    train_data_categorical_columns = ['route',
                                      'time_slot',
                                      'bus_crowd',
                                      'available_buses',
                                      'type_of_bus',
                                      'day_name',
                                      'is_Weekday',
                                      'day_type',
                                      'is_holiday',
                                      'trafic',
                                      'available_bus_drivers',
                                      'weather',
                                      'route_condition',
                                      'busy_reason']

    new_colums = ['x0_Kegalle - Colombo', 'x0_Kurunegala - Colombo', 'x0_Mathara - Colombo', 'x0_Negambo - Colombo',
                  'x0_Ragama - Colombo', 'x0_Vavuniaya - Colombo', 'x0_colombo - Kegalle', 'x0_colombo - Kurunegala',
                  'x0_colombo - Negambo', 'x0_colombo - Ragama', 'x0_colombo - Vavuniaya', 'x0_colombo - matara',
                  'x1_10.00 - 12.00', 'x1_12.00 - 14.00', 'x1_14.00 - 16.00', 'x1_16.00 - 18.00', 'x1_18.00 - 20.00',
                  'x1_20.00 - 22.00', 'x1_22.00 - 24.00', 'x1_6.00 - 8.00', 'x1_8.00 - 10.00', 'x2_high', 'x2_low',
                  'x2_medium', 'x3_0', 'x3_1', 'x3_2', 'x3_3', 'x3_4', 'x3_5', 'x3_6', 'x4_AC', 'x4_normal',
                  'x5_Friday', 'x5_Monday', 'x5_Saturday', 'x5_Sunday', 'x5_Thursday', 'x5_Tuesday', 'x5_Wednesday',
                  'x6_no', 'x6_yes', 'x7_Christmas Day', 'x7_Deepavali', 'x7_Sinhala and Tamil New Year',
                  'x7_Thai Pongal', 'x7_Vesak', 'x7_normal', 'x7_poya', 'x8_no', 'x8_yes', 'x9_high', 'x9_low',
                  'x9_medium', 'x9_no', 'x10_enough', 'x10_no', 'x11_normal', 'x11_rain', 'x12_good', 'x12_slippery',
                  'x13_normal', 'x13_office', 'x13_office,school', 'x13_school', 'x13_school,normal']

    route = int_features[0]
    time_slot = int_features[1]
    bus_crowd = int_features[2]
    available_buses = int_features[3]
    type_of_bus = int_features[4]
    day_name = int_features[5]
    is_Weekday = int_features[6]
    day_type = int_features[7]
    is_holiday = int_features[8]
    trafic = int_features[9]
    available_bus_drivers = int_features[10]
    weather = int_features[11]
    route_condition = int_features[12]
    busy_reason = int_features[13]

    data = [{'route': route, 'time_slot': time_slot, 'bus_crowd': bus_crowd, 'available_buses': int(available_buses),
             'type_of_bus': type_of_bus, 'day_name': day_name, 'is_Weekday': is_Weekday, 'day_type': day_type,
             'is_holiday': is_holiday, 'trafic': trafic, 'available_bus_drivers': available_bus_drivers,
             'weather': weather, 'route_condition': route_condition, 'busy_reason': busy_reason}]

    try_df = pd.DataFrame(data)
    try_df = pd.DataFrame(busEncoder.transform(try_df[train_data_categorical_columns]).toarray(), columns=new_colums)
    y_predict = tree_clf.predict(try_df)
    x = y_predict.tolist()
    y = x[0]
    data = y
    return render_template('index.html', prediction_text1='Predicted Bus Schedule : {}'.format(data))


# debug is set to True in development environment and set to False in production environment
if __name__ == "__main__":
    app.run(debug=False)
