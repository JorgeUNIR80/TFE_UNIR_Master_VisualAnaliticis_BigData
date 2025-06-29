import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import pandas as pd
import joblib
from binance.client import Client
from tensorflow.keras.models import load_model
#import time
from datetime import datetime, timezone, timedelta



# Parámetros
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR
sequence_length = 60
umbral_compra = 60000
file_path = "/home/ojupisha/predicciones_btc_hour.csv"

# Inicializar cliente Binance
API_KEY = '2POZsiboHQwbAIH4RcMxNu2ZSSxNC2eSG1rEDIbQNIrGDZ6EVkxFwa04oFfs3ACw'
API_SECRET = 'YJGcEy8XPyzsVjKH2GrGCRmxRp2Gg0klvJNwtPO8e4Mjl3SleXyKHLmwCiUi1aRW'
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'
client._sync_request_time = True

# Cargar modelo y scaler
model = load_model("/home/ojupisha/modelo_btc_lstm.h5")
scaler = joblib.load("/home/ojupisha/scaler.pkl")

# Obtener datos recientes
def get_recent_data():
    klines = client.get_klines(symbol=symbol, interval=interval, limit=sequence_length)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    cols_to_float = ['open', 'high', 'low', 'close', 'volume']
    df[cols_to_float] = df[cols_to_float].astype(float)
    return df

# Predicción y guardado
def predict_next_price():
    df = get_recent_data()
    scaled = scaler.transform(df[['close']].values)
    X_input = scaled[-sequence_length:].reshape(1, sequence_length, 1)
    prediction = model.predict(X_input)
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    return predicted_price

def get_price_now():
    df = get_recent_data()
    return df['close'].values[-1]

def compare_and_store(predicted_price, real_price, timestamp):
    # Leer la predicción anterior (si existe)
    if os.path.exists(file_path):
        df_hist = pd.read_csv(file_path)
        if not df_hist.empty:
            prediccion_anterior = df_hist.iloc[-1]['Prediccion_BTC']
            porcentaje_error = abs((real_price - prediccion_anterior) / real_price) * 100
        else:
            prediccion_anterior = predicted_price
            porcentaje_error = 0.0  # No hay dato anterior
    else:
        prediccion_anterior = predicted_price
        porcentaje_error = 0.0  # Primer registro

    # Determinar dirección del mercado con respecto a la predicción actual
    mercado = "Baja" if real_price > predicted_price else "Sube"

    # Formato legible para la hora
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    # Crear línea para guardar
    new_line = f"{time_str},{real_price:.2f},{predicted_price:.2f},{porcentaje_error:.2f},{mercado}\n"

    # Guardar en el archivo
    write_header = not os.path.exists(file_path)
    with open(file_path, "a") as f:
        if write_header:
            f.write("Fecha_Hora,Precio_BTC,Prediccion_BTC,Porcentaje_error,Mercado\n")
        f.write(new_line)

    # Imprimir resumen
    print(f"[{time_str}] Real: {real_price:.2f}, Predicción: {predicted_price:.2f}, Error: {porcentaje_error:.2f}%, Mercado: {mercado}")



def hourly_prediction_loop():

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    predicted_price = predict_next_price()
    print(f"[{now}] Predicción realizada: {predicted_price:.2f}")

    real_price = get_price_now()
    compare_and_store(predicted_price, real_price, datetime.now(timezone.utc))


# Ejecutar
if __name__ == "__main__":
    hourly_prediction_loop()
