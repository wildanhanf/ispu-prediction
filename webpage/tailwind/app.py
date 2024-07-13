from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from keras.models import load_model  

app = Flask(__name__)

df = pd.read_csv('clean_data.csv')
df['tanggal'] = pd.to_datetime(df['tanggal'])

df_copy = df.copy()
df_copy.set_index('tanggal', inplace=True)

def categorize_ispu(ispu):
    if ispu <= 50:
        return 'Baik'
    elif ispu <= 100:
        return 'Sedang'
    elif ispu <= 200:
        return 'Tidak Sehat'
    elif ispu <= 300:
        return 'Sangat Tidak Sehat'
    else:
        return 'Berbahaya'

@app.route('/', methods=['GET'])
def index():
    aktual_labels = []
    aktual_data = []
    aktual_list = []

    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['tanggal'] = df['tanggal'].dt.strftime('%Y-%m-%d')

    aktual_list = df.values.tolist()
    aktual_labels = [row[0] for row in aktual_list]
    aktual_data = [row[1] for row in aktual_list]
    return render_template('tentang.html', aktual_list=aktual_list, aktual_labels=aktual_labels, aktual_data=aktual_data)

@app.route('/ispu', methods=['GET', 'POST'])
def predict():
    selected_date = None
    sequence = None
    future_df = pd.DataFrame()
    future_data_for_table = []
    future_data = []
    labels = []
    data = []
    test_labels = []
    test_data = []

    if request.method == 'POST':
        date_string = request.form['date-input']
        date_object = datetime.strptime(date_string, "%m/%d/%Y")
        selected_date = date_object.strftime("%Y-%m-%d")
        selected_date = pd.to_datetime(selected_date)

        last_date = pd.to_datetime('2023-11-30')
        sequence = (selected_date - last_date).days

        # Run the LSTM model with the calculated sequence
        future_df, test_data_for_chart = run_lstm_model(sequence)
        future_data_for_table = future_df.sort_values(by='tanggal', ascending=False).head(5)
        future_data_for_table = future_data_for_table.values.tolist()

        future_data = future_df.values.tolist()
        combined_data = test_data_for_chart + future_data
        labels =  [row[0] for row in combined_data]
        data = [row[1] for row in combined_data]
        
        test_labels = [row[0] for row in test_data_for_chart]
        test_data = [row[1] for row in test_data_for_chart]
    return render_template('ispu.html', selected_date=selected_date, sequence=sequence, future_data_for_table=future_data_for_table, labels=labels, data=data, test_labels=test_labels, test_data=test_data)

def run_lstm_model(sequence):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['pm_duakomalima'].values.reshape(-1,1))

    # Membagi data menjadi data training dan testing
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:train_size, :]
    test_data = scaled_data[train_size:, :]

    # Fungsi untuk membuat dataset
    def create_dataset(dataset, time_steps):
        X, Y = [], []
        for i in range(len(dataset)-time_steps):
            a = dataset[i:(i+time_steps), 0]
            X.append(a)
            Y.append(dataset[i + time_steps, 0])
        return np.array(X), np.array(Y)

    time_steps = 10

    X_train, Y_train = create_dataset(train_data, time_steps)
    X_test, Y_test = create_dataset(test_data, time_steps)

    # Reshape input menjadi [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = load_model('lstm_pm25_model.h5')

    # Mengambil data terakhir dari set pengujian sebagai titik awal untuk prediksi masa depan
    last_data = test_data[-time_steps:]

    # Membuat array untuk menyimpan prediksi masa depan
    future_predictions = []

    for _ in range(sequence):
        # Mengambil subset terakhir dari last_data dengan panjang time_steps
        input_data = last_data[-time_steps:].reshape(1, 1, time_steps)
        
        # Membuat prediksi
        prediction = model.predict(input_data)
        
        # Menyimpan prediksi
        future_predictions.append(prediction[0, 0])
        
        # Memperbarui last_data dengan menambahkan prediksi terbaru
        last_data = np.append(last_data, prediction)
        
    # Inverse transformasi prediksi masa depan
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Membulatkan hasil prediksi
    future_predictions = np.round(future_predictions)

    # Membuat DataFrame untuk hasil prediksi masa depan
    future_dates = pd.date_range(start=df_copy.index[-1], periods=sequence + 1, inclusive='right')
    future_df = pd.DataFrame(future_predictions, columns=['predicted_ispu'])
    future_df['tanggal'] = future_dates
    future_df['category'] = future_df['predicted_ispu'].apply(categorize_ispu)
    future_df = future_df[['tanggal', 'predicted_ispu', 'category']] 
    future_df.reset_index(drop=True, inplace=True) 
    future_df['tanggal'] = future_df['tanggal'].dt.strftime('%Y-%m-%d')

    test_dates = df_copy.index[time_steps+time_steps+len(X_train):time_steps+time_steps+len(X_train)+len(X_test)]
    test_data_for_chart = pd.DataFrame(scaler.inverse_transform(Y_test.reshape(-1, 1)), columns=['pm_duakomalima'])
    test_data_for_chart['tanggal'] = test_dates
    test_data_for_chart['tanggal'] = pd.to_datetime(test_data_for_chart['tanggal'])
    test_data_for_chart['tanggal'] = test_data_for_chart['tanggal'].dt.strftime('%Y-%m-%d')
    test_data_for_chart['category'] = test_data_for_chart['pm_duakomalima'].apply(categorize_ispu)
    test_data_for_chart = test_data_for_chart[['tanggal', 'pm_duakomalima', 'category']] 
    test_data_for_chart = test_data_for_chart.values.tolist()

    return future_df, test_data_for_chart

if __name__ == '__main__':
    app.run(debug=True)
