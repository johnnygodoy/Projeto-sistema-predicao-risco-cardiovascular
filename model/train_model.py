# ================================
# IMPORTS
# ================================
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import os
import shutil
import seaborn as sns
import random
import json
import time
from datetime import datetime


from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import (classification_report, confusion_matrix,accuracy_score,
    f1_score, precision_score, recall_score,roc_auc_score, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline


# ================================
# CONFIG ASSETS
# ================================
ASSETS_DIR = "assets"

# arquivos que PODEM ser apagados (gerados)
FILES_TO_RESET = [
    "calibration_curve.png",
    "confusion_matrix.png",
    "feature_importance.png",
    "precision_recall.png",
    "roc_curve.png",
    "shap_summary.png",
    "ga_results.json",
    "model_comparison.json",
    "ga_convergence.png"
]

def reset_generated_images():
    for file in FILES_TO_RESET:
        path = os.path.join(ASSETS_DIR, file)
        if os.path.exists(path):
            os.remove(path)

def save_plot(filename):
    path = os.path.join(ASSETS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# limpa SOMENTE os gráficos gerados
reset_generated_images()

# ================================
# CONFIG ALGORITMO GENÉTICO
# ================================
GA_RESULTS_FILE = os.path.join(ASSETS_DIR, "ga_results.json")
MODEL_COMPARISON_FILE = os.path.join(ASSETS_DIR, "model_comparison.json")

XGB_PARAM_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3, 0.5],
    "reg_lambda": [0.5, 1, 1.5, 2]
}

# ================================
# LOAD
# ================================
df = pd.read_csv("data/cardio_train.csv", sep=";")

if "id" in df.columns:
    df = df.drop(columns=["id"])


# ================================
# FEATURE ENGINEERING
# ================================
df["age_years"] = df["age"] / 365

# ================================
# LIMPEZA
# ================================
df = df[(df["age_years"] >= 18) & (df["age_years"] <= 100)]
df = df[(df["height"] >= 140) & (df["height"] <= 210)]
df = df[(df["weight"] >= 40) & (df["weight"] <= 200)]
df = df[(df["ap_hi"] >= 90) & (df["ap_hi"] <= 200)]
df = df[(df["ap_lo"] >= 60) & (df["ap_lo"] <= 120)]

# ================================
# NOVAS FEATURES
# ================================
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df = df[(df["bmi"] >= 15) & (df["bmi"] <= 50)]
df["pressure_age"] = df["ap_hi"] * df["age_years"]

# ================================
# FEATURES FINAIS
# ================================
features_final = [
    "ap_hi", "ap_lo", "age_years", "bmi",
    "cholesterol", "gluc", "smoke",
    "alco", "active", "pressure_age"
]

X = df[features_final]
y = df["cardio"]


# ================================
# SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split interno para o Algoritmo Genético.
# O X_test continua reservado para avaliação final.
X_train_ga, X_valid_ga, y_train_ga, y_valid_ga = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# ================================
# PIPELINE
# ================================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, X.columns)
])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

def criar_pipeline_xgboost(parametros=None):
    if parametros is None:
        parametros = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }

    return Pipeline([
        ("prep", preprocessor),
        ("model", XGBClassifier(
            **parametros,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])


# ================================
# FUNÇÕES DO ALGORITMO GENÉTICO
# ================================
def encontrar_melhor_threshold(y_true, y_prob, recall_minimo=0.80):
    melhor_threshold = 0.5
    melhor_f1 = -1

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_temp = (y_prob > threshold).astype(int)

        recall = recall_score(y_true, y_pred_temp, zero_division=0)
        f1 = f1_score(y_true, y_pred_temp, zero_division=0)

        if recall >= recall_minimo and f1 > melhor_f1:
            melhor_f1 = f1
            melhor_threshold = threshold

    if melhor_f1 == -1:
        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred_temp = (y_prob > threshold).astype(int)
            f1 = f1_score(y_true, y_pred_temp, zero_division=0)

            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_threshold = threshold

    return round(float(melhor_threshold), 2)


def criar_individuo():
    return {
        parametro: random.choice(valores)
        for parametro, valores in XGB_PARAM_SPACE.items()
    }


def criar_populacao(tamanho_populacao):
    return [criar_individuo() for _ in range(tamanho_populacao)]


def avaliar_individuo(individuo):
    modelo = criar_pipeline_xgboost(individuo)

    modelo.fit(X_train_ga, y_train_ga)

    y_prob_valid = modelo.predict_proba(X_valid_ga)[:, 1]

    threshold = encontrar_melhor_threshold(
        y_valid_ga,
        y_prob_valid,
        recall_minimo=0.80
    )

    y_pred_valid = (y_prob_valid > threshold).astype(int)

    metricas = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_valid_ga, y_pred_valid)),
        "precision": float(precision_score(y_valid_ga, y_pred_valid, zero_division=0)),
        "recall": float(recall_score(y_valid_ga, y_pred_valid, zero_division=0)),
        "f1_score": float(f1_score(y_valid_ga, y_pred_valid, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_valid_ga, y_prob_valid))
    }

    # Função fitness principal.
    # No contexto médico, usamos F1 para equilibrar precision e recall.
    fitness = metricas["f1_score"]

    return fitness, metricas


def selecao_torneio(populacao_avaliada, tamanho_torneio=3):
    tamanho_torneio = min(tamanho_torneio, len(populacao_avaliada))

    competidores = random.sample(populacao_avaliada, tamanho_torneio)

    competidores = sorted(
        competidores,
        key=lambda item: item["fitness"],
        reverse=True
    )

    return competidores[0]["individuo"]


def cruzamento(pai1, pai2, taxa_cruzamento):
    if random.random() > taxa_cruzamento:
        return pai1.copy(), pai2.copy()

    filho1 = {}
    filho2 = {}

    for gene in pai1.keys():
        if random.random() < 0.5:
            filho1[gene] = pai1[gene]
            filho2[gene] = pai2[gene]
        else:
            filho1[gene] = pai2[gene]
            filho2[gene] = pai1[gene]

    return filho1, filho2


def mutacao(individuo, taxa_mutacao):
    individuo_mutado = individuo.copy()

    for gene in individuo_mutado.keys():
        if random.random() < taxa_mutacao:
            individuo_mutado[gene] = random.choice(XGB_PARAM_SPACE[gene])

    return individuo_mutado


def executar_algoritmo_genetico(
    nome_experimento,
    tamanho_populacao,
    geracoes,
    taxa_cruzamento,
    taxa_mutacao
):
    inicio = time.time()

    populacao = criar_populacao(tamanho_populacao)

    melhor_individuo_global = None
    melhor_fitness_global = -1
    melhores_metricas_global = None

    historico = []

    print("\n" + "=" * 80)
    print(nome_experimento)
    print("=" * 80)

    for geracao in range(geracoes):
        populacao_avaliada = []

        print(f"\nGeração {geracao + 1}/{geracoes}")

        for individuo in populacao:
            fitness, metricas = avaliar_individuo(individuo)

            populacao_avaliada.append({
                "individuo": individuo,
                "fitness": fitness,
                "metricas": metricas
            })

            if fitness > melhor_fitness_global:
                melhor_fitness_global = fitness
                melhor_individuo_global = individuo.copy()
                melhores_metricas_global = metricas.copy()

        populacao_avaliada = sorted(
            populacao_avaliada,
            key=lambda item: item["fitness"],
            reverse=True
        )

        melhor_geracao = populacao_avaliada[0]

        historico.append({
            "geracao": geracao + 1,
            "melhor_fitness": float(melhor_geracao["fitness"]),
            "melhor_individuo": melhor_geracao["individuo"],
            "metricas": melhor_geracao["metricas"]
        })

        print(f"Melhor fitness da geração: {melhor_geracao['fitness']:.4f}")
        print(f"Melhor indivíduo: {melhor_geracao['individuo']}")
        print(f"Métricas: {melhor_geracao['metricas']}")

        nova_populacao = []

        # Elitismo: preserva o melhor indivíduo da geração.
        nova_populacao.append(melhor_geracao["individuo"])

        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_torneio(populacao_avaliada)
            pai2 = selecao_torneio(populacao_avaliada)

            filho1, filho2 = cruzamento(pai1, pai2, taxa_cruzamento)

            filho1 = mutacao(filho1, taxa_mutacao)
            filho2 = mutacao(filho2, taxa_mutacao)

            nova_populacao.append(filho1)

            if len(nova_populacao) < tamanho_populacao:
                nova_populacao.append(filho2)

        populacao = nova_populacao

    tempo_total = time.time() - inicio

    return {
        "nome_experimento": nome_experimento,
        "melhor_individuo": melhor_individuo_global,
        "melhor_fitness": float(melhor_fitness_global),
        "melhores_metricas_validacao": melhores_metricas_global,
        "tamanho_populacao": tamanho_populacao,
        "geracoes": geracoes,
        "taxa_cruzamento": taxa_cruzamento,
        "taxa_mutacao": taxa_mutacao,
        "tempo_total_segundos": round(tempo_total, 2),
        "historico": historico
    }


def executar_3_experimentos_ga():
    experimentos = [
        {
            "nome": "Experimento 1 - População pequena e mutação baixa",
            "tamanho_populacao": 6,
            "geracoes": 3,
            "taxa_cruzamento": 0.8,
            "taxa_mutacao": 0.05
        },
        {
            "nome": "Experimento 2 - População média e mutação moderada",
            "tamanho_populacao": 8,
            "geracoes": 4,
            "taxa_cruzamento": 0.8,
            "taxa_mutacao": 0.10
        },
        {
            "nome": "Experimento 3 - População maior e mutação alta",
            "tamanho_populacao": 10,
            "geracoes": 5,
            "taxa_cruzamento": 0.9,
            "taxa_mutacao": 0.20
        }
    ]

    resultados = []

    for experimento in experimentos:
        resultado = executar_algoritmo_genetico(
            nome_experimento=experimento["nome"],
            tamanho_populacao=experimento["tamanho_populacao"],
            geracoes=experimento["geracoes"],
            taxa_cruzamento=experimento["taxa_cruzamento"],
            taxa_mutacao=experimento["taxa_mutacao"]
        )

        resultados.append(resultado)

    resultados = sorted(
        resultados,
        key=lambda item: item["melhor_fitness"],
        reverse=True
    )

    with open(GA_RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(resultados, file, indent=4, ensure_ascii=False)

    return resultados


# ================================
# MODELO ORIGINAL PARA COMPARAÇÃO
# ================================
print("\nTreinando modelo original para comparação...")

pipeline_original = criar_pipeline_xgboost()
pipeline_original.fit(X_train, y_train)

y_prob_original = pipeline_original.predict_proba(X_test)[:, 1]

threshold_original = encontrar_melhor_threshold(
    y_test,
    y_prob_original,
    recall_minimo=0.80
)

y_pred_original = (y_prob_original > threshold_original).astype(int)

metricas_original = {
    "modelo": "XGBoost Original",
    "threshold": threshold_original,
    "accuracy": float(accuracy_score(y_test, y_pred_original)),
    "precision": float(precision_score(y_test, y_pred_original, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_original, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred_original, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob_original))
}

print("\nMétricas do modelo original:")
print(json.dumps(metricas_original, indent=4, ensure_ascii=False))

# ================================
# EXECUÇÃO DO ALGORITMO GENÉTICO
# ================================
print("\nExecutando Algoritmo Genético...")

resultados_ga = executar_3_experimentos_ga()

melhor_resultado_ga = resultados_ga[0]
melhores_parametros = melhor_resultado_ga["melhor_individuo"]

print("\nMelhores hiperparâmetros encontrados pelo Algoritmo Genético:")
print(melhores_parametros)

# Treina o modelo final usando os melhores hiperparâmetros encontrados.
pipeline = criar_pipeline_xgboost(melhores_parametros)
pipeline.fit(X_train, y_train)

# ================================
# THRESHOLD OTIMIZADO
# ================================
y_prob = pipeline.predict_proba(X_test)[:, 1]

best_threshold = 0.5
best_precision = 0

for t in np.arange(0.1, 0.9, 0.01):
    y_pred_temp = (y_prob > t).astype(int)

    recall = recall_score(y_test, y_pred_temp, zero_division=0)
    precision = precision_score(y_test, y_pred_temp, zero_division=0)

    if recall >= 0.80 and precision > best_precision:
        best_precision = precision
        best_threshold = t

print(f"Melhor threshold: {best_threshold:.2f}")


# ================================
# RESULTADOS
# ================================
y_pred = (y_prob > best_threshold).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

metricas_otimizado = {
    "modelo": "XGBoost Otimizado por Algoritmo Genético",
    "threshold": float(round(best_threshold, 2)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "melhores_hiperparametros": melhores_parametros,
    "melhor_experimento": melhor_resultado_ga["nome_experimento"]
}

comparativo = {
    "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modelo_original": metricas_original,
    "modelo_otimizado_algoritmo_genetico": metricas_otimizado,
    "experimentos_algoritmo_genetico": resultados_ga
}

with open(MODEL_COMPARISON_FILE, "w", encoding="utf-8") as file:
    json.dump(comparativo, file, indent=4, ensure_ascii=False)



# ================================
# CONVERGÊNCIA DO ALGORITMO GENÉTICO
# ================================
plt.figure(figsize=(10, 6))

for resultado in resultados_ga:
    geracoes = [item["geracao"] for item in resultado["historico"]]
    fitness = [item["melhor_fitness"] for item in resultado["historico"]]

    plt.plot(geracoes, fitness, marker="o", label=resultado["nome_experimento"])

plt.title("Convergência do Algoritmo Genético")
plt.xlabel("Geração")
plt.ylabel("Melhor Fitness - F1-score")
plt.legend()
save_plot("ga_convergence.png")

# ================================
# ROC
# ================================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("Curva ROC")
plt.legend()
save_plot("roc_curve.png")


# ================================
# PRECISION-RECALL
# ================================
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8,6))
plt.plot(recall_vals, precision_vals)
plt.title("Precision-Recall")
save_plot("precision_recall.png")


# ================================
# MATRIZ DE CONFUSÃO
# ================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
save_plot("confusion_matrix.png")


# ================================
# CALIBRATION
# ================================
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure(figsize=(8,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], linestyle='--')
plt.title("Calibration Curve")
save_plot("calibration_curve.png")


# ================================
# FEATURE IMPORTANCE
# ================================
xgb_model = pipeline.named_steps["model"]

importances = xgb_model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Importância das Features")
save_plot("feature_importance.png")


# ================================
# SHAP
# ================================
X_test_transformed = pipeline.named_steps["prep"].transform(X_test)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=X.columns,
    show=False
)

plt.savefig(os.path.join(ASSETS_DIR, "shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close()


# ================================
# SAVE MODEL
# ================================
with open("model/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Modelo salvo com sucesso!")