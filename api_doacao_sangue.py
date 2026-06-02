"""
DoaVida — API de Machine Learning
Stack  : FastAPI + scikit-learn + joblib (sem PyCaret)
Python : 3.13 compatível
Porta  : 8001
Docs   : http://localhost:8001/docs
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ── Carregar modelos ──────────────────────────────────────────
def load_models():
    clf   = joblib.load(os.path.join(MODELS_DIR, "clf_retorno_doador.pkl"))
    reg   = joblib.load(os.path.join(MODELS_DIR, "reg_volume_doacao.pkl"))
    clust = joblib.load(os.path.join(MODELS_DIR, "clust_segmentacao_rfm.pkl"))
    sim   = joblib.load(os.path.join(MODELS_DIR, "rec_campanha_similaridade.pkl"))
    lista = joblib.load(os.path.join(MODELS_DIR, "rec_lista_campanhas.pkl"))
    with open(os.path.join(MODELS_DIR, "metadados_modelos.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return clf, reg, clust, sim, lista, meta

try:
    clf_bundle, reg_bundle, clust_bundle, sim_campanhas, lista_campanhas, metadados = load_models()
    MODELS_LOADED = True
    print(f"Modelos carregados — versão {metadados['versao']}")
except Exception as e:
    print(f"[AVISO] Modelos não encontrados: {e}")
    print("[AVISO] Execute 'python train.py' primeiro.")
    MODELS_LOADED = False
    metadados = {}

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="DoaVida — API de Machine Learning",
    description="Predição de retorno, segmentação RFMT e recomendação de campanhas.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────
class DoadorInput(BaseModel):
    recencia_meses:              float = Field(..., ge=0, description="Meses desde última doação")
    frequencia_doacoes:          float = Field(..., ge=0, description="Total de doações realizadas")
    volume_total_cc:             float = Field(..., ge=0, description="Volume total doado (c.c.)")
    tempo_desde_primeira_doacao: float = Field(..., ge=0, description="Meses desde 1ª doação")
    risco_inatividade:           Optional[str] = Field("Atencao", description="Ativo|Atencao|Em_Risco|Inativo")

class RecomendacaoInput(BaseModel):
    campanhas_participadas: List[str] = Field(default=[], description="Campanhas que o doador já participou")
    n_recomendacoes:        int       = Field(3, ge=1, le=8)

# ── Utilitários ───────────────────────────────────────────────
RISCO_MAP = {"Ativo": 0, "Atencao": 1, "Em_Risco": 2, "Inativo": 3}

def preparar_features(dados: dict, features: list) -> np.ndarray:
    """Constrói e enriquece features a partir do input do doador."""
    tempo = max(dados["tempo_desde_primeira_doacao"], 1)
    freq  = max(dados["frequencia_doacoes"], 1)

    enriquecido = {
        "recencia_meses":              dados["recencia_meses"],
        "frequencia_doacoes":          dados["frequencia_doacoes"],
        "volume_total_cc":             dados["volume_total_cc"],
        "tempo_desde_primeira_doacao": dados["tempo_desde_primeira_doacao"],
        "taxa_doacao_mensal":          dados["frequencia_doacoes"] / tempo,
        "volume_medio_por_doacao":     dados["volume_total_cc"]    / freq,
        "score_engajamento":           min(
            0.6 * (1 - dados["recencia_meses"] / 74.4) +
            0.4 * (dados["frequencia_doacoes"] / 50.0), 1.0
        ),
        "recencia_quadratica":         dados["recencia_meses"] ** 2,
        "log_volume_total":            np.log1p(dados["volume_total_cc"]),
        "risco_inatividade_num":       RISCO_MAP.get(dados.get("risco_inatividade", "Atencao"), 1),
    }
    return np.array([[enriquecido[f] for f in features]])

def check_models():
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Modelos não carregados. Execute 'python train.py' e reinicie a API."
        )

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    return {
        "status":             "online" if MODELS_LOADED else "sem_modelos",
        "modelos_carregados": MODELS_LOADED,
        "versao":             metadados.get("versao", "—"),
        "n_doadores_treino":  metadados.get("n_doadores_treino", 0),
        "metricas":           metadados.get("metricas", {}),
        "instrucao":          None if MODELS_LOADED else "Execute 'python train.py' para treinar os modelos",
    }

@app.post("/predizer/retorno", tags=["Agendamento"])
def predizer_retorno(doador: DoadorInput):
    """Probabilidade de o doador retornar para uma campanha (classificação)."""
    check_models()
    try:
        clf    = clf_bundle["model"]
        scaler = clf_bundle["scaler"]
        feats  = clf_bundle["features"]

        X      = preparar_features(doador.dict(), feats)
        X_s    = scaler.transform(X)
        proba  = float(clf.predict_proba(X_s)[0][1])
        classe = int(clf.predict(X_s)[0])

        return {
            "vai_retornar":          bool(classe == 1),
            "probabilidade_retorno": round(proba, 4),
            "nivel_prioridade":      "alta" if proba > 0.7 else "media" if proba > 0.4 else "baixa",
            "recomendacao":          "Notificar para agendamento" if classe == 1 else "Incluir em campanha de reativação",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predizer/volume", tags=["Estoque"])
def predizer_volume(doador: DoadorInput):
    """Estima o volume de sangue (c.c.) que o doador irá contribuir (regressão)."""
    check_models()
    try:
        reg    = reg_bundle["model"]
        scaler = reg_bundle["scaler"]
        feats  = reg_bundle["features"]

        X      = preparar_features(doador.dict(), feats)
        X_s    = scaler.transform(X)
        log_vol = float(reg.predict(X_s)[0])
        vol_cc  = float(np.expm1(log_vol))

        return {
            "volume_estimado_cc":     round(max(vol_cc, 0), 1),
            "volume_estimado_litros": round(max(vol_cc, 0) / 1000, 3),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segmentar/doador", tags=["Campanhas"])
def segmentar_doador(doador: DoadorInput):
    """Classifica o doador no segmento RFMT (VIP, Frequente, Inativo, etc.)."""
    check_models()
    try:
        clust  = clust_bundle["model"]
        scaler = clust_bundle["scaler"]
        feats  = clust_bundle["features"]

        X      = preparar_features(doador.dict(), feats)
        X_s    = scaler.transform(X)
        cluster_num = int(clust.predict(X_s)[0])
        cluster_key = f"Cluster {cluster_num}"

        return {
            "cluster":  cluster_key,
            "segmento": metadados.get("segmentos", {}).get(cluster_key, "Perfil Regular"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recomendar/campanhas", tags=["Campanhas"])
def recomendar_campanhas(payload: RecomendacaoInput):
    """Recomenda campanhas com base no histórico (Item-Item Cosine Similarity)."""
    check_models()
    try:
        participou = payload.campanhas_participadas
        nao_viu    = [c for c in lista_campanhas if c not in participou]

        if not nao_viu:
            return {"recomendacoes": [], "mensagem": "Doador já participou de todas as campanhas"}

        scores = {}
        for camp in nao_viu:
            validos = [p for p in participou if p in sim_campanhas.columns]
            if validos:
                scores[camp] = float(sim_campanhas.loc[camp, validos].mean())
            else:
                scores[camp] = float(sim_campanhas.loc[camp].mean())

        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:payload.n_recomendacoes]
        return {
            "recomendacoes": [
                {"rank": i + 1, "campanha": c, "score_relevancia": round(s, 4)}
                for i, (c, s) in enumerate(top)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
