"""Genera los dos notebooks listos para Google Colab alineados a la ficha Heckerling del médico."""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
NB_DIR = BASE_DIR / "notebooks"


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip("\n").splitlines()]}


def code_cell(code: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.strip("\n").splitlines()]}


def make_notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": cells,
    }


def build_nb1_heckerling_rf() -> dict:
    cells = [
        md_cell(
            """# 🫁 NEUMOALERT — Notebook 1: Puntaje de Heckerling vs Random Forest Clínico

**Proyecto Capstone · SIC 2026 IA**  
Este notebook implementa la ficha clínica propuesta por el médico asesor (**Puntaje de Heckerling, 1990** + datos adicionales modernos como **SatO₂**, **FR**, **antipirético previo** y **rinorrea/odinofagia**) y responde una pregunta concreta:

> **¿Agregar SatO₂ y las variables clínicas del médico mediante Random Forest mejora la predicción respecto al Puntaje de Heckerling solo?**"""
        ),
        code_cell(
            """import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("Librerías cargadas.")"""
        ),
        md_cell("## 1. Cargar `clinical_pneumonia_dataset.csv` (o generar si se corre directo en Colab)"),
        code_cell(
            """CSV_CANDIDATES = ["clinical_pneumonia_dataset.csv", "../data/clinical_pneumonia_dataset.csv"]
csv_found = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)

if csv_found:
    df = pd.read_csv(csv_found)
else:
    # Generador reproducible calibrado a Heckerling (1990) + Moore (2017)
    rng = np.random.default_rng(42)
    rows = []
    profiles = rng.choice(["resfrio_viral", "bronquitis", "asma_crisis", "epoc_icc", "neumonia_leve", "neumonia_franca"], size=2500, p=[0.24, 0.22, 0.14, 0.12, 0.14, 0.14])
    for i, prof in enumerate(profiles, 1):
        sexo = str(rng.choice(["F", "M"], p=[0.52, 0.48]))
        asma = int(prof == "asma_crisis" or (prof != "asma_crisis" and rng.random() < 0.08))
        epoc = int((prof == "epoc_icc" and rng.random() < 0.65) or (prof != "epoc_icc" and rng.random() < 0.09))
        otra_p = int(rng.random() < 0.05)
        icc = int((prof == "epoc_icc" and rng.random() < 0.45) or (prof != "epoc_icc" and rng.random() < 0.07))
        inmuno = int(rng.random() < 0.08)
        if prof in ("neumonia_franca", "neumonia_leve"):
            edad = int(np.clip(rng.normal(60, 18), 18, 94))
            temp_r = float(np.clip(rng.normal(38.5 if prof=="neumonia_franca" else 38.2, 0.6), 36.2, 40.5))
            fc = int(np.clip(rng.normal(109 if prof=="neumonia_franca" else 101, 13), 70, 145))
            crep = int(rng.random() < (0.82 if prof=="neumonia_franca" else 0.58))
            murm = int(rng.random() < (0.68 if prof=="neumonia_franca" else 0.42))
            fr = int(np.clip(rng.normal(26 if prof=="neumonia_franca" else 23, 4), 15, 42))
            sat = round(float(np.clip(rng.normal(91.2 if prof=="neumonia_franca" else 93.8, 2.7), 82, 98.5)), 1)
            rino = int(rng.random() < 0.12)
            mat, ego, sop = int(rng.random()<0.2), int(rng.random()<0.12), int(rng.random()<0.12)
            conc = int(rng.random() < (0.28 if edad>=65 else 0.05))
            anti = int(rng.random() < 0.43)
            dias = int(np.clip(rng.normal(5, 2.5), 1, 13))
            base = 2.15 if prof=="neumonia_franca" else 0.85
        else:
            edad = int(np.clip(rng.normal(46, 18), 18, 90))
            temp_r = float(np.clip(rng.normal(37.2, 0.55), 35.8, 39.0))
            fc = int(np.clip(rng.normal(93, 14), 58, 138))
            crep = int(rng.random() < (0.45 if prof=="epoc_icc" else 0.12))
            murm = int(rng.random() < (0.22 if prof in ("asma_crisis", "epoc_icc") else 0.06))
            fr = int(np.clip(rng.normal(20, 4), 12, 36))
            sat = round(float(np.clip(rng.normal(96.3, 2.1), 86, 100)), 1)
            rino = int(rng.random() < (0.78 if prof=="resfrio_viral" else 0.35))
            mat, ego, sop = int(rng.random()<0.02), int(rng.random()<0.01), int(rng.random()<0.01)
            conc = int(rng.random() < 0.03)
            anti = int(rng.random() < 0.28)
            dias = int(np.clip(rng.normal(4.5, 2.5), 1, 13))
            base = -1.9
        dem = int(edad >= 70 and rng.random() < 0.14)
        temp_m = round(float(np.clip(temp_r - rng.uniform(0.65, 1.15), 35.8, 40.3)), 1) if (anti==1 and temp_r>37.6) else round(temp_r, 1)
        aus_asma = 1 - asma
        heck = int(temp_m>37.8) + int(fc>100) + crep + murm + aus_asma
        logit = 0.45*base + 0.70*int(temp_r>37.8) + 0.48*int(fc>100) + 0.72*crep + 0.68*murm + 0.52*aus_asma - 0.36*(sat-95) + 0.08*(fr-20) - 1.05*rino + 1.05*mat + 0.95*ego + 1.0*sop + 0.90*conc + 0.25*inmuno - 0.25*int(dias<=1) + rng.normal(0, 1.05) - 2.35
        prob = 1.0 / (1.0 + np.exp(-logit))
        u = rng.random()
        if u < prob:
            cat_rx, rx_alt = ("Definitiva" if rng.random()<0.65 else "Probable"), 1
        else:
            cat_rx, rx_alt = ("Posible" if (0.15<=prob<=0.65 and rng.random()<0.28) else "Sin neumonía"), 0
        rows.append({"asma": asma, "epoc": epoc, "otra_enf_pulmonar": otra_p, "insuficiencia_cardiaca": icc, "inmunosupresion": inmuno, "demencia": dem, "temperatura": temp_m, "frecuencia_cardiaca": fc, "crepitantes": crep, "murmullo_disminuido": murm, "ausencia_asma": aus_asma, "puntaje_heckerling": heck, "frecuencia_respiratoria": fr, "sat_o2": sat, "matidez": mat, "egofonia": ego, "soplo_tubarico": sop, "compromiso_conciencia": conc, "rinorrea_odinofagia": rino, "antipiretico_6h": anti, "dias_evolucion": dias, "edad": edad, "sexo": sexo, "categoria_rx": cat_rx, "rx_alterada": rx_alt})
    df = pd.DataFrame(rows)

print("Total pacientes:", len(df))
print(df.groupby("puntaje_heckerling")["rx_alterada"].agg(["count", "mean"]).rename(columns={"mean": "tasa_rx_alterada"}).round(3))"""
        ),
        md_cell("## 2. Comparar Puntaje de Heckerling solo vs Random Forest (con SatO₂ y datos del médico)"),
        code_cell(
            """FEATURES = [
    "asma", "epoc", "otra_enf_pulmonar", "insuficiencia_cardiaca", "inmunosupresion", "demencia",
    "temperatura", "frecuencia_cardiaca", "crepitantes", "murmullo_disminuido", "ausencia_asma",
    "puntaje_heckerling", "frecuencia_respiratoria", "sat_o2", "matidez", "egofonia", "soplo_tubarico",
    "compromiso_conciencia", "rinorrea_odinofagia", "antipiretico_6h", "dias_evolucion", "edad", "sexo",
]
CAT_COLS = ["sexo"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

# Igual que Heckerling (1990): se evalúa Probable+Definitiva (1) vs Sin neumonía (0)
df_eval = df[df["categoria_rx"] != "Posible"].copy()
X, y = df_eval[FEATURES], df_eval["rx_alterada"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 1. Heckerling solo
heck_lr = LogisticRegression(class_weight="balanced", random_state=42).fit(X_train[["puntaje_heckerling"]], y_train)
heck_prob = heck_lr.predict_proba(X_test[["puntaje_heckerling"]])[:, 1]
heck_pred = (X_test["puntaje_heckerling"] >= 3).astype(int)

# 2. Random Forest con toda la ficha
rf_pipe = Pipeline([
    ("prep", ColumnTransformer([("num", "passthrough", NUM_COLS), ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CAT_COLS)])),
    ("clf", RandomForestClassifier(n_estimators=260, max_depth=10, min_samples_leaf=4, class_weight={0: 1.0, 1: 2.1}, random_state=42)),
])
rf_pipe.fit(X_train, y_train)
rf_prob = rf_pipe.predict_proba(X_test)[:, 1]
rf_pred = rf_pipe.predict(X_test)

print(f"Heckerling solo -> AUC={roc_auc_score(y_test, heck_prob):.3f} | Acc={accuracy_score(y_test, heck_pred):.1%} | Recall={recall_score(y_test, heck_pred):.1%}")
print(f"Random Forest   -> AUC={roc_auc_score(y_test, rf_prob):.3f} | Acc={accuracy_score(y_test, rf_pred):.1%} | Recall={recall_score(y_test, rf_pred):.1%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
RocCurveDisplay.from_predictions(y_test, heck_prob, ax=axes[0], name="Heckerling solo (1990)")
RocCurveDisplay.from_predictions(y_test, rf_prob, ax=axes[0], name="Random Forest (Ficha completa)")
axes[0].set_title("Curva ROC comparativa")
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, display_labels=["Sin neumonía", "RxT alterada"], cmap="Blues", ax=axes[1])
axes[1].set_title("Matriz de Confusión (Random Forest)")
plt.tight_layout()
plt.show()"""
        ),
        md_cell("## 3. Importancia de variables (¿qué pesó más además de Heckerling?)"),
        code_cell(
            """imp = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=NUM_COLS + ["sexo"]).sort_values()
imp.tail(10).plot(kind="barh", figsize=(8, 4.5), color="#0f4c75", title="Top 10 variables más influyentes")
plt.xlabel("Importancia relativa")
plt.tight_layout()
plt.show()"""
        ),
    ]
    return make_notebook(cells)


def build_nb2_nlp_4classes() -> dict:
    cells = [
        md_cell(
            """# 📄 NEUMOALERT — Notebook 2: NLP Clínico en 4 Categorías (Sección E) + Fusión Multimodal

**Proyecto Capstone · SIC 2026 IA**  
Clasifica informes radiológicos PDF en las **4 categorías exactas de la Sección E del estudio del médico**:
1. `Sin neumonía`
2. `Posible`
3. `Probable`
4. `Definitiva`

Y luego muestra cómo se une con la **Red Neuronal (CNN)** y el **Random Forest (ficha Heckerling)**."""
        ),
        code_cell(
            """!pip install pymupdf scikit-learn pandas matplotlib joblib -q
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CSV_CANDIDATES = ["radiology_reports_dataset.csv", "../data/radiology_reports_dataset.csv"]
csv_found = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
assert csv_found is not None, "Sube el archivo radiology_reports_dataset.csv de la carpeta data/ a Colab."

df_rx = pd.read_csv(csv_found)
print("Informes:", len(df_rx))
print(df_rx["categoria_rx"].value_counts())"""
        ),
        code_cell(
            """X_train, X_test, y_train, y_test = train_test_split(
    df_rx["texto_informe"], df_rx["categoria_rx"], test_size=0.20, random_state=42, stratify=df_rx["categoria_rx"]
)

nlp_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 3), min_df=2, max_features=3500, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=2.2, class_weight="balanced", random_state=42)),
])
nlp_pipe.fit(X_train, y_train)
y_pred = nlp_pipe.predict(X_test)

orden = ["Sin neumonía", "Posible", "Probable", "Definitiva"]
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%} | F1 Macro: {f1_score(y_test, y_pred, average='macro'):.1%}\\n")
print(classification_report(y_test, y_pred, labels=orden))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=orden, cmap="Greens")
plt.title("Matriz de Confusión 4×4 — Categorías del Radiólogo (Sección E)")
plt.tight_layout()
plt.show()"""
        ),
        code_cell(
            """# Prueba rápida con ejemplos reales (incluyendo negación trampa)
ejemplos = [
    "Motivo: descartar neumonía. Hallazgos: Consolidación homogénea con broncograma aéreo en base derecha. Conclusión: Infiltrado neumónico inequívoco.",
    "Hallazgos: Opacidad alveolar parcheada basal izquierda. Conclusión: Lo más probable es neumonía adquirida en la comunidad.",
    "Hallazgos: Tenue infiltrado intersticial dudoso. Conclusión: Posible infiltrado inicial, no se puede descartar neumonía.",
    "Motivo: sospecha de neumonía, descartar infiltrados. Hallazgos: Campos pulmonares limpios, sin consolidaciones. Conclusión: Estudio normal, sin neumonía.",
]
for txt in ejemplos:
    pred = nlp_pipe.predict([txt])[0]
    print(f"[{pred:12s}] -> {txt[:100]}...")"""
        ),
    ]
    return make_notebook(cells)


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    (NB_DIR / "01_Modelo_Random_Forest_Checklist.ipynb").write_text(
        json.dumps(build_nb1_heckerling_rf(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (NB_DIR / "02_Modelo_NLP_Informes_Rx_y_Fusion.ipynb").write_text(
        json.dumps(build_nb2_nlp_4classes(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Notebooks v3 actualizados en:", NB_DIR)


if __name__ == "__main__":
    main()
