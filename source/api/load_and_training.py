import io

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math

from database import save_model_to_db, init_db, KerasModel


def load_data(company, start_date):
    # Parâmetros
    # symbol = 'PETR4.SA'
    # start_date = '2005-01-01'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Coleta dos dados
    df = yf.download(company, start=start_date, end=end_date)

    # Coluna de fechamento de valores, removendo valores vazios/nulos
    df = df[['Close']].dropna()

    # Pré-processamento
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df)

    # Função para criar sequências
    def create_sequences(data, seq_length):
        x2, y2 = [], []
        for i in range(seq_length, len(data)):
            x2.append(data[i - seq_length:i, 0])
            y2.append(data[i, 0])
        return np.array(x2), np.array(y2)

    x, y = create_sequences(data_scaled, 60)
    last_sequence = x[-1]

    x = x.reshape((x.shape[0], x.shape[1], 1))

    # 80% dados de teste
    split = int(0.8 * len(x))
    X_train, X_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    # Modelo sequencial
    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training
    # history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluations
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
    real = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = math.sqrt(mean_squared_error(real, predicted))
    mae = mean_absolute_error(real, predicted)

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

    engine = init_db()

    save_model_to_db(model=model, company=company, engine=engine, scaler=scaler, last_sequence=last_sequence)

