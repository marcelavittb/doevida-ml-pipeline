"""
DoaVida — Treinamento dos modelos ML
Stack: scikit-learn + pandas + numpy (sem PyCaret)
Python: 3.13 compatível
Execute: python train.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, r2_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

print("=" * 55)
print(" DoaVida — Treinamento dos Modelos ML")
print("=" * 55)

# ── 1. Carregar dataset ───────────────────────────────────────
print("\n[1/6] Carregando dataset UCI Blood Transfusion...")

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
COLUNAS = {
    "Recency (months)":                             "recencia_meses",
    "Frequency (times)":                            "frequencia_doacoes",
    "Monetary (c.c. blood)":                        "volume_total_cc",
    "Time (months)":                                "tempo_desde_primeira_doacao",
    "whether he/she donated blood in March 2007":   "vai_doar",
}

try:
    df_raw = pd.read_csv(URL)
    df_raw.rename(columns=COLUNAS, inplace=True)
except Exception:
    print("  UCI indisponível, usando fallback...")
    df_raw = pd.read_csv(
        "https://raw.githubusercontent.com/dsrscientist/dataset1/master/blood_transfusion.csv"
    )
    df_raw.columns = list(COLUNAS.values())

df_raw = df_raw.drop_duplicates()
print(f"  {len(df_raw)} doadores carregados.")

# ── 2. Engenharia de features ────────────────────────────────
print("\n[2/6] Criando features de negócio...")

df = df_raw.copy()

df["taxa_doacao_mensal"]      = df["frequencia_doacoes"] / df["tempo_desde_primeira_doacao"].replace(0, 1)
df["volume_medio_por_doacao"] = df["volume_total_cc"]    / df["frequencia_doacoes"].replace(0, 1)

rec_norm  = 1 - (df["recencia_meses"]    / df["recencia_meses"].max())
freq_norm =      df["frequencia_doacoes"] / df["frequencia_doacoes"].max()
df["score_engajamento"]   = (0.6 * rec_norm + 0.4 * freq_norm).round(4)
df["recencia_quadratica"] = df["recencia_meses"] ** 2
df["log_volume_total"]    = np.log1p(df["volume_total_cc"])

# Risco de inatividade (categórico → numérico)
bins   = [-1, 3, 9, 18, 1000]
labels = ["Ativo", "Atencao", "Em_Risco", "Inativo"]
df["risco_inatividade"]    = pd.cut(df["recencia_meses"], bins=bins, labels=labels)
le = LabelEncoder().fit(labels)
df["risco_inatividade_num"] = le.transform(df["risco_inatividade"].astype(str))

print("  Features criadas com sucesso.")

# ── 3. Pipeline de Classificação ─────────────────────────────
print("\n[3/6] Treinando classificador (retorno do doador)...")

FEATURES_CLF = [
    "recencia_meses", "frequencia_doacoes", "volume_total_cc",
    "tempo_desde_primeira_doacao", "taxa_doacao_mensal",
    "volume_medio_por_doacao", "score_engajamento",
    "recencia_quadratica", "risco_inatividade_num",
]

X_clf = df[FEATURES_CLF].fillna(df[FEATURES_CLF].median())
y_clf = df["vai_doar"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Normalizar
scaler_clf = StandardScaler()
X_train_s  = scaler_clf.fit_transform(X_train)
X_test_s   = scaler_clf.transform(X_test)

# SMOTE para desbalanceamento
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_s, y_train)

clf_model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42
)
clf_model.fit(X_train_sm, y_train_sm)

auc = roc_auc_score(y_test, clf_model.predict_proba(X_test_s)[:, 1])
print(f"  AUC-ROC no teste: {auc:.4f}")

# ── 4. Pipeline de Regressão ──────────────────────────────────
print("\n[4/6] Treinando regressor (volume de doação)...")

FEATURES_REG = [
    "recencia_meses", "frequencia_doacoes", "tempo_desde_primeira_doacao",
    "taxa_doacao_mensal", "volume_medio_por_doacao",
    "score_engajamento", "recencia_quadratica",
]

X_reg = df[FEATURES_REG].fillna(df[FEATURES_REG].median())
y_reg = df["log_volume_total"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
X_train_rs = scaler_reg.fit_transform(X_train_r)
X_test_rs  = scaler_reg.transform(X_test_r)

reg_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, random_state=42
)
reg_model.fit(X_train_rs, y_train_r)

r2 = r2_score(y_test_r, reg_model.predict(X_test_rs))
print(f"  R² no teste: {r2:.4f}")

# ── 5. Pipeline de Clusterização ─────────────────────────────
print("\n[5/6] Treinando clusterizador (segmentação RFMT)...")

FEATURES_CLUST = [
    "recencia_meses", "frequencia_doacoes", "volume_total_cc",
    "tempo_desde_primeira_doacao", "taxa_doacao_mensal", "score_engajamento",
]

X_clust = df[FEATURES_CLUST].fillna(df[FEATURES_CLUST].median())
scaler_clust = StandardScaler()
X_clust_s    = scaler_clust.fit_transform(X_clust)

# Selecionar melhor K
best_k, best_sil = 3, -1
for k in range(2, 7):
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_clust_s)
    sil = silhouette_score(X_clust_s, lbl)
    print(f"  K={k}  Silhouette={sil:.4f}")
    if sil > best_sil:
        best_sil, best_k = sil, k

print(f"  Melhor K: {best_k} (Silhouette={best_sil:.4f})")

clust_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster_num"] = clust_model.fit_predict(X_clust_s)
df["cluster"]     = df["cluster_num"].apply(lambda x: f"Cluster {x}")

# Nomear clusters
perfil        = df.groupby("cluster")[FEATURES_CLUST].mean()
medias_globais = df[FEATURES_CLUST].mean()

def nomear_cluster(row):
    baixa_rec = row["recencia_meses"]     < medias_globais["recencia_meses"]
    alta_freq = row["frequencia_doacoes"] > medias_globais["frequencia_doacoes"]
    alto_vol  = row["volume_total_cc"]    > medias_globais["volume_total_cc"]
    alto_eng  = row["score_engajamento"]  > medias_globais["score_engajamento"]
    if baixa_rec and alta_freq and alto_vol: return "Doador VIP"
    elif baixa_rec and alto_eng:             return "Doador Frequente"
    elif not baixa_rec and alta_freq:        return "Em Risco de Inatividade"
    elif not baixa_rec and not alta_freq:    return "Doador Inativo"
    elif alta_freq and not alto_vol:         return "Doador Regular"
    else:                                    return "Doador Novato"

nomes_clusters = {c: nomear_cluster(perfil.loc[c]) for c in perfil.index}
df["segmento"] = df["cluster"].map(nomes_clusters)
print("  Segmentos:", dict(df["segmento"].value_counts()))

# ── 6. Sistema de Recomendação ───────────────────────────────
print("\n[6/6] Construindo sistema de recomendação de campanhas...")

np.random.seed(42)

CAMPANHAS = [
    "Campanha_Urgencia_O_Negativo",
    "Campanha_Doacao_Regular_Mensal",
    "Campanha_Semana_do_Doador",
    "Campanha_Doacao_Hospitais_Publicos",
    "Campanha_Tipo_A_Positivo",
    "Campanha_Reativacao_Inativos",
    "Campanha_Primeira_Doacao",
    "Campanha_Doador_VIP_Exclusiva",
]

PROB_BASE = {
    "Campanha_Urgencia_O_Negativo":       0.20,
    "Campanha_Doacao_Regular_Mensal":     0.30,
    "Campanha_Semana_do_Doador":          0.25,
    "Campanha_Doacao_Hospitais_Publicos": 0.20,
    "Campanha_Tipo_A_Positivo":           0.15,
    "Campanha_Reativacao_Inativos":       0.10,
    "Campanha_Primeira_Doacao":           0.15,
    "Campanha_Doador_VIP_Exclusiva":      0.10,
}

def prob_participacao(segmento, campanha):
    p = PROB_BASE.get(campanha, 0.15)
    if "VIP"      in segmento and ("VIP_Exclusiva" in campanha or "Regular" in campanha): p *= 3.0
    elif "Frequente" in segmento and ("Regular"    in campanha or "Semana"  in campanha): p *= 2.5
    elif "Inativo"   in segmento:
        p = p * 4.0 if "Reativacao" in campanha else p * 0.3
    elif "Novato"    in segmento and "Primeira" in campanha: p *= 3.5
    return min(p, 1.0)

participacoes = np.array([
    [1 if np.random.rand() < prob_participacao(df.iloc[i]["segmento"], c) else 0
     for c in CAMPANHAS]
    for i in range(len(df))
])

df_participacao  = pd.DataFrame(participacoes, columns=CAMPANHAS, index=df.index)
campanha_sim     = cosine_similarity(df_participacao.T)
campanha_sim_df  = pd.DataFrame(campanha_sim, index=CAMPANHAS, columns=CAMPANHAS)

print(f"  Matriz de similaridade: {campanha_sim_df.shape}")

# ── Salvar tudo ──────────────────────────────────────────────
print("\nSalvando modelos...")
os.makedirs("models", exist_ok=True)
VERSAO = datetime.now().strftime("%Y%m%d_%H%M")

joblib.dump({"model": clf_model,   "scaler": scaler_clf,   "features": FEATURES_CLF},   "models/clf_retorno_doador.pkl")
joblib.dump({"model": reg_model,   "scaler": scaler_reg,   "features": FEATURES_REG},   "models/reg_volume_doacao.pkl")
joblib.dump({"model": clust_model, "scaler": scaler_clust, "features": FEATURES_CLUST}, "models/clust_segmentacao_rfm.pkl")
joblib.dump(campanha_sim_df,  "models/rec_campanha_similaridade.pkl")
joblib.dump(CAMPANHAS,        "models/rec_lista_campanhas.pkl")

metadados = {
    "versao":            VERSAO,
    "data_treino":       datetime.now().isoformat(),
    "n_doadores_treino": int(len(df)),
    "dataset":           "UCI Blood Transfusion Service Center",
    "metricas": {
        "classificacao_auc": round(auc, 4),
        "regressao_r2":      round(r2,  4),
        "clusterizacao_silhouette": round(best_sil, 4),
        "n_clusters": best_k,
    },
    "segmentos": nomes_clusters,
    "campanhas": CAMPANHAS,
}

with open("models/metadados_modelos.json", "w", encoding="utf-8") as f:
    json.dump(metadados, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 55)
print(" Modelos salvos em models/")
print("=" * 55)
for arq in sorted(os.listdir("models")):
    tam = os.path.getsize(f"models/{arq}") / 1024
    print(f"  {arq:<45} {tam:>8.1f} KB")
print(f"\nVersão: {VERSAO}")
print(f"AUC-ROC: {auc:.4f}  |  R²: {r2:.4f}  |  Silhouette: {best_sil:.4f}")
print("\nPróximo passo:")
print("  uvicorn api_doacao_sangue:app --host 0.0.0.0 --port 8001 --reload")
