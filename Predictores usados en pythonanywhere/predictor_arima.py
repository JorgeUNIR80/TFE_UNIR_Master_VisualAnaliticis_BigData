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
from datetime import datetime, timezone, timedelta
import ta  # Importar la librería TA-Lib, instalar con pip install ta
from statsmodels.tsa.arima.model import ARIMA  # Importar ARIMA
from sklearn.metrics import mean_squared_error  # Importar para comparar modelos

# Parámetros
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1HOUR
sequence_length = 60
umbral_compra = 60000
file_path = "/home/ojupisha/predicciones_btc_hour_arima.csv"
arima_order = (2, 1, 3)  # Ejemplo de orden ARIMA (p, d, q) - ¡AJUSTAR ESTO!

# Inicializar cliente Binance
API_KEY = '2POZsiboHQwbAIH4RcMxNu2ZSSxNC2eSG1rEDIbQNIrGDZ6EVkxFwa04oFfs3ACw'
API_SECRET = 'YJGcEy8XPyzsVjKH2GrGCRmxRp2Gg0klvJNwtPO8e4Mjl3SleXyKHLmwCiUi1aRW'
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'
client._sync_request_time = True

# Cargar modelo y scaler
model_lstm = load_model("/home/ojupisha/modelo_btc_lstm.h5")
scaler = joblib.load("/home/ojupisha/scaler.pkl")

# Obtener datos recientes
def get_recent_data():
    klines = client.get_klines(symbol=symbol, interval=interval, limit=sequence_length + 100)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    cols_to_float = ['open', 'high', 'low', 'close', 'volume']
    df[cols_to_float] = df[cols_to_float].astype(float)
    return df

# Calcular indicadores técnicos
def calculate_indicators(df):
    # Simple Moving Average (SMA)
    df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)

    # Exponential Moving Average (EMA)
    df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)

    # Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)

    # Moving Average Convergence Divergence (MACD)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']

    # Parabolic SAR
    df['SAR'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()

    return df

# Predicción y guardado
def predict_next_price():
    df = get_recent_data()
    df = calculate_indicators(df)
    scaled = scaler.transform(df[['close']].values)
    X_input = scaled[-sequence_length:].reshape(1, sequence_length, 1)
    predicted_price_lstm = model_lstm.predict(X_input)[0][0]
    predicted_price_lstm = scaler.inverse_transform([[predicted_price_lstm]])[0][0]

    # Preparar datos para ARIMA
    history = df['close'].values
    model_arima = ARIMA(history, order=arima_order)
    model_arima_fit = model_arima.fit()
    predicted_price_arima = model_arima_fit.forecast(steps=1)[0]

    return predicted_price_lstm, predicted_price_arima, df.iloc[-1]

def get_price_now():
    df = get_recent_data()
    return df['close'].values[-1], df.iloc[-1]

def compare_and_store(predicted_price_lstm, predicted_price_arima, real_price, timestamp, indicators):
    # Leer la predicción anterior (si existe)
    if os.path.exists(file_path):
        df_hist = pd.read_csv(file_path)
        if not df_hist.empty:
            prediccion_anterior_lstm = df_hist.iloc[-1]['Prediccion_LSTM']  # Cambiado el nombre
            prediccion_anterior_arima = df_hist.iloc[-1]['Prediccion_ARIMA']
            porcentaje_error_lstm = abs((real_price - prediccion_anterior_lstm) / real_price) * 100
            porcentaje_error_arima = abs((real_price - prediccion_anterior_arima) / real_price) * 100
        else:
            prediccion_anterior_lstm = predicted_price_lstm
            prediccion_anterior_arima = predicted_price_arima
            porcentaje_error_lstm = 0.0
            porcentaje_error_arima = 0.0
    else:
        prediccion_anterior_lstm = predicted_price_lstm
        prediccion_anterior_arima = predicted_price_arima
        porcentaje_error_lstm = 0.0
        porcentaje_error_arima = 0.0

    # Determinar dirección del mercado con respecto a la predicción actual
    mercado_lstm = "Baja" if real_price > predicted_price_lstm else "Sube"
    mercado_arima = "Baja" if real_price > predicted_price_arima else "Sube"

    # Formato legible para la hora
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    # Crear línea para guardar, incluyendo indicadores
    new_line = f"{time_str},{real_price:.2f},{predicted_price_lstm:.2f},{porcentaje_error_lstm:.2f},{mercado_lstm}," \
               f"{predicted_price_arima:.2f},{porcentaje_error_arima:.2f},{mercado_arima},"

    # Add indicators if they exist
    if 'SMA_20' in indicators:
        new_line += f"{indicators['SMA_20']:.2f},"
    else:
        new_line += "NaN,"
    if 'SMA_50' in indicators:
        new_line += f"{indicators['SMA_50']:.2f},"
    else:
        new_line += "NaN,"
    if 'EMA_20' in indicators:
        new_line += f"{indicators['EMA_20']:.2f},"
    else:
        new_line += "NaN,"
    if 'EMA_50' in indicators:
        new_line += f"{indicators['EMA_50']:.2f},"
    else:
        new_line += "NaN,"
    if 'RSI' in indicators:
        new_line += f"{indicators['RSI']:.2f},"
    else:
        new_line += "NaN,"
    if 'MACD' in indicators:
        new_line += f"{indicators['MACD']:.2f},"
    else:
        new_line += "NaN,"
    if 'MACD_signal' in indicators:
        new_line += f"{indicators['MACD_signal']:.2f},"
    else:
        new_line += "NaN,"
    if 'MACD_hist' in indicators:
        new_line += f"{indicators['MACD_hist']:.2f},"
    else:
        new_line += "NaN,"
    if 'BB_upper' in indicators:
        new_line += f"{indicators['BB_upper']:.2f},"
    else:
        new_line += "NaN,"
    if 'BB_mid' in indicators:
        new_line += f"{indicators['BB_mid']:.2f},"
    else:
        new_line += "NaN,"
    if 'BB_lower' in indicators:
        new_line += f"{indicators['BB_lower']:.2f},"
    else:
        new_line += "NaN,"
    if 'BB_width' in indicators:
        new_line += f"{indicators['BB_width']:.2f},"
    else:
        new_line += "NaN,"
    if 'SAR' in indicators:
        new_line += f"{indicators['SAR']:.2f}"
    else:
        new_line += "NaN"
    new_line += "\n"

    # Guardar en el archivo
    write_header = not os.path.exists(file_path)
    with open(file_path, "a") as f:
        if write_header:
            f.write("Fecha_Hora,Precio_BTC,Prediccion_LSTM,Porcentaje_error_LSTM,Mercado_LSTM,Prediccion_ARIMA,Porcentaje_error_ARIMA,Mercado_ARIMA,SMA_20,SMA_50,EMA_20,EMA_50,RSI,MACD,MACD_signal,MACD_hist,BB_upper,BB_mid,BB_lower,BB_width,SAR\n")
        f.write(new_line)

    # Imprimir resumen
    print(
        f"[{time_str}] Real: {real_price:.2f}, Predicción LSTM: {predicted_price_lstm:.2f}, Error LSTM: {porcentaje_error_lstm:.2f}%, Mercado LSTM: {mercado_lstm}, "
        f"Predicción ARIMA: {predicted_price_arima:.2f}, Error ARIMA: {porcentaje_error_arima:.2f}%, Mercado ARIMA: {mercado_arima}"
    )
    indicator_str = ", ".join([f"{indicator}={indicators.get(indicator, 'NaN'):.2f}" for indicator in
                             ['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                              'BB_upper', 'BB_mid', 'BB_lower', 'BB_width', 'SAR']])
    print(f"Indicadores: {indicator_str}")



def hourly_prediction_loop():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    predicted_price_lstm, predicted_price_arima, latest_data = predict_next_price()
    print(f"[{now}] Predicción LSTM: {predicted_price_lstm:.2f}, Predicción ARIMA: {predicted_price_arima:.2f}")

    real_price, real_data = get_price_now()
    compare_and_store(predicted_price_lstm, predicted_price_arima, real_price, datetime.now(timezone.utc), real_data)

    # Estrategia de ejemplo (esto es solo un ejemplo, NO es asesoramiento financiero)
    if predicted_price_lstm > real_price and real_data['RSI'] < 30:
        print("💡 Señal de Compra (LSTM): Predicción alcista y RSI sobrevendido")
    elif predicted_price_lstm < real_price and real_data['RSI'] > 70:
        print("💡 Señal de Venta (LSTM): Predicción bajista y RSI sobrecomprado")
    elif predicted_price_arima > real_price and real_data['RSI'] < 30:
        print("💡 Señal de Compra (ARIMA): Predicción alcista y RSI sobrevendido")
    elif predicted_price_arima < real_price and real_data['RSI'] > 70:
        print("💡 Señal de Venta (ARIMA): Predicción bajista y RSI sobrecomprado")
    elif abs(predicted_price_lstm - real_price) / real_price < 0.01 and abs(
            predicted_price_arima - real_price) / real_price < 0.01:
        print("Neutral: Ambos modelos predicen un cambio de precio muy pequeño. Esperar.")
    elif predicted_price_lstm > real_price and predicted_price_arima > real_price:
        print("💡 Señal de Compra: Ambos modelos predicen subida.")
    elif predicted_price_lstm < real_price and predicted_price_arima < real_price:
        print("💡 Señal de Venta: Ambos modelos predicen bajada.")
    else:
        print("Sin señal clara: Los modelos no coinciden en la dirección.")

# Ejecutar
if __name__ == "__main__":
    hourly_prediction_loop()
