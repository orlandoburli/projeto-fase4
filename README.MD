# 📈 Previsão de Ações da Petrobras usando LSTM

Este projeto desenvolve um modelo de rede neural LSTM para prever o valor de fechamento das ações da Petrobrás (PETR4.SA), utilizando dados históricos desde 2005.

## 🚀 Funcionalidades
- Coleta de dados do Yahoo Finance.
- Treinamento de rede neural LSTM.
- API REST com FastAPI para realizar previsões.

## 🏗️ Tecnologias
- Python
- TensorFlow / Keras
- FastAPI
- Docker
- yFinance
- scikit-learn

## 🔧 Como Executar Localmente

### Clone o repositório:
```bash

git clone https://github.com/orlandoburli/projeto-fase4.git
cd projeto-fase4/infra
docker compose up -d --build

```

Acesse em: http://127.0.0.1:8000/

### Colete e treine um modelo:
```bash
curl --location 'http://localhost:8000/collect-and-train' \
--header 'Content-Type: application/json' \
--data '{
    "Company": "PETR4.SA",
    "StartDate": "2015-01-01"
}'
```

### Realize uma predição nesse modelo:

```bash
curl --location 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "company": "PETR4.SA",
    "daysToPredict": "10"
}'
```