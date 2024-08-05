# Prediksi Indeks Standar Pencemar Udara di DKI Jakarta Berdasarkan PM2,5 Menggunakan Long Short Term Memory

Proyek ini bertujuan untuk memprediksi ISPU berdasarkan PM2,5 di DKI Jakarta menggunakan model *Long Short-Term Memory* (LSTM). PM2,5 adalah partikel polutan udara yang sangat kecil dan berbahaya bagi kesehatan manusia. Dengan memprediksi konsentrasi PM2,5, pemerintah dapat mengambil langkah-langkah pencegahan yang tepat untuk mengurangi dampak negatif polusi udara terhadap kesehatan masyarakat dan lingkungan. Serta dapat menjadi pengetahuan bagi masyarakat terhadap resiko dari polusi udara disekitarnya.

## Library yang digunakan:
Dengan Python 3.11.9
- Pandas 2.2.2
- Matplotlib 3.8.4
- NumPy 1.26.4
- Scikit-learn 1.4.2
- Keras 3.3.3
- TensorFlow 2.16.1

## Hyperparameter yang akan diuji
Pelatihan model akan menggunakan 1000 epochs, batch_size bernilai 32, time_steps bernilai 10, dan EarlyStopping. Setiap kombinasi hyperparameter akan dilatih masing-masing sebanyak 5 kali percobaan. 
| *Hyperparameter*  | Nilai               |
| ----------------- | -----------------   |
| Units             | 50, 100, 150        |
| Jumlah layer      | 1, 2                |
| Fungsi aktivasi   | tanh, relu, sigmoid |
| Optimizer         | adam, rmsprop       |
| Learning rate     | 0,001; 0,01         |

## Hasil Model Terbaik

| Units | Layers | Activation | Optimizer | Learning Rate | RMSE     | MAPE   |
|-------|--------|------------|-----------|---------------|----------|--------|
| 150   | 1      | tanh       | adam      | 0.01          | 21.39767 | 12.20% |

## Demo Implementasi Prediksi Pada Antarmuka
Pastikan path pada github\ispu-prediction\webpage\tailwind

```
python app.py
```
