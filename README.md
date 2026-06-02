# DoaVida — API de Machine Learning

**Python 3.13 compatível — sem PyCaret**

## Setup (só na primeira vez)

Abra o PowerShell ou Prompt de Comando na pasta do projeto:

```cmd
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Treinar os modelos (~2 min)

```cmd
python train.py
```

Deve aparecer no final:
```
Modelos salvos em models/
AUC-ROC: 0.78xx  |  R²: 0.94xx  |  Silhouette: 0.xx
```

## Subir a API

```cmd
uvicorn api_doacao_sangue:app --host 0.0.0.0 --port 8001 --reload
```

Acesse: http://localhost:8001/docs

## Testar

```cmd
curl http://localhost:8001/
```

Resposta esperada:
```json
{"status": "online", "modelos_carregados": true, "versao": "...", ...}
```

## Retreinar (quando tiver dados novos)

```cmd
python train.py
```
Reinicie a API depois.
