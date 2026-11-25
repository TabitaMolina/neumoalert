"""
PASO 2 · Entrenamiento y comparación científica de modelos de NEUMOALERT:

1. Comparación cara a cara en el mismo conjunto de prueba (Test Set no visto):
   - Modelo A (Referencia clásica): Puntaje de Heckerling solo (0 a 5 puntos)
   - Modelo B (Línea base ML): Regresión Logística con Heckerling + variables del médico
   - Modelo C (Modelo principal): Random Forest con Heckerling + variables del médico (SatO2,
     FR, antipirético últimas 6h, rinorrea/odinofagia, matidez, egofonía, soplo tubárico,
     compromiso de conciencia, enfermedades previas, edad, sexo y días de evolución)
2. Modelo NLP Clínico (TF-IDF 1-3 gramas + Regresión Logística) en las 4 categorías
   exactas de la Sección E: Sin neumonía, Posible, Probable, Definitiva.

Guarda los pipelines entrenados en `models/` y todas las métricas comparativas en `models/model_metrics.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.clinical_rf import CLINICAL_FEATURES, FEATURE_LABELS_ES

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def train_and_compare_clinical() -> dict:
    df = pd.read_csv(DATA_DIR / "clinical_pneumonia_dataset.csv")

    # Igual que el estudio original de Heckerling (1990):
    # el desenlace principal excluye las radiografías dudosas ("Posible") para entrenar limpio,
    # comparando "Probable + Definitiva" (1) vs "Sin neumonía" (0).
    df_eval = df[df["categoria_rx"] != "Posible"].copy().reset_index(drop=True)
    X = df_eval[CLINICAL_FEATURES].copy()
    y = df_eval["rx_alterada"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- A. Referencia: Puntaje de Heckerling solo (0 a 5) ---
    heck_only_model = LogisticRegression(class_weight="balanced", random_state=42)
    heck_only_model.fit(X_train[["puntaje_heckerling"]], y_train)
    heck_prob = heck_only_model.predict_proba(X_test[["puntaje_heckerling"]])[:, 1]
    # Umbral práctico tradicional: puntaje >= 3 (o prob >= 0.5)
    heck_pred = (X_test["puntaje_heckerling"] >= 3).astype(int)
    heck_metrics = {
        "nombre": "Puntaje de Heckerling solo (1990)",
        "accuracy": round(float(accuracy_score(y_test, heck_pred)), 4),
        "precision": round(float(precision_score(y_test, heck_pred)), 4),
        "recall": round(float(recall_score(y_test, heck_pred)), 4),
        "f1": round(float(f1_score(y_test, heck_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, heck_prob)), 4),
    }

    cat_cols = ["sexo"]
    num_cols = [c for c in CLINICAL_FEATURES if c not in cat_cols]

    # --- B. Regresión Logística con ficha completa ---
    lr_pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), num_cols),
                        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight={0: 1.0, 1: 1.8}, random_state=42)),
        ]
    )
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
    lr_metrics = {
        "nombre": "Regresión Logística (Ficha completa)",
        "accuracy": round(float(accuracy_score(y_test, lr_pred)), 4),
        "precision": round(float(precision_score(y_test, lr_pred)), 4),
        "recall": round(float(recall_score(y_test, lr_pred)), 4),
        "f1": round(float(f1_score(y_test, lr_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, lr_prob)), 4),
    }

    # --- C. Random Forest Clínico (Ficha completa: Heckerling + datos adicionales del médico) ---
    rf_pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", num_cols),
                        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=260,
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight={0: 1.0, 1: 2.1},  # prioriza alta sensibilidad (recall)
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_roc = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_rec = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="recall")

    rf_pipe.fit(X_train, y_train)
    rf_pred = rf_pipe.predict(X_test)
    rf_prob = rf_pipe.predict_proba(X_test)[:, 1]

    feature_names_out = list(num_cols) + ["sexo"]
    importances = rf_pipe.named_steps["classifier"].feature_importances_
    fi_list = sorted(
        [
            {
                "feature": fname,
                "label": FEATURE_LABELS_ES.get(fname, fname),
                "importance": round(float(imp), 4),
            }
            for fname, imp in zip(feature_names_out, importances)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # Tabla empírica por Puntaje de Heckerling (0 a 5) sobre todo el dataset
    tabla_heckerling = []
    for score in range(6):
        sub = df[df["puntaje_heckerling"] == score]
        tabla_heckerling.append(
            {
                "puntaje": score,
                "pacientes": int(len(sub)),
                "pct_alterada": round(float(sub["rx_alterada"].mean() * 100), 1) if len(sub) else 0.0,
            }
        )

    rf_metrics = {
        "n_total_dataset": int(len(df)),
        "n_excluidos_posible": int((df["categoria_rx"] == "Posible").sum()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "prevalencia_rx_alterada": round(float(y.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, rf_pred)), 4),
        "precision": round(float(precision_score(y_test, rf_pred)), 4),
        "recall": round(float(recall_score(y_test, rf_pred)), 4),
        "f1": round(float(f1_score(y_test, rf_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, rf_prob)), 4),
        "cv_roc_auc_mean": round(float(np.mean(cv_roc)), 4),
        "cv_recall_mean": round(float(np.mean(cv_rec)), 4),
        "confusion_matrix": confusion_matrix(y_test, rf_pred).tolist(),
        "labels": ["Sin neumonía (0)", "RxT alterada (1)"],
        "feature_importances": fi_list,
        "comparacion_modelos": [
            heck_metrics,
            lr_metrics,
            {
                "nombre": "Random Forest (Heckerling + Datos extra)",
                "accuracy": round(float(accuracy_score(y_test, rf_pred)), 4),
                "precision": round(float(precision_score(y_test, rf_pred)), 4),
                "recall": round(float(recall_score(y_test, rf_pred)), 4),
                "f1": round(float(f1_score(y_test, rf_pred)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, rf_prob)), 4),
            },
        ],
        "tabla_heckerling": tabla_heckerling,
    }

    joblib.dump(
        {
            "pipeline": rf_pipe,
            "features": CLINICAL_FEATURES,
            "feature_importances": fi_list,
            "metrics": rf_metrics,
        },
        MODELS_DIR / "rf_clinical_model.joblib",
    )

    print("=== COMPARACIÓN CLÍNICA (TEST SET NO VISTO) ===")
    for m in rf_metrics["comparacion_modelos"]:
        print(f"  {m['nombre']:42s} | AUC={m['roc_auc']:.4f} | Acc={m['accuracy']:.1%} | Recall={m['recall']:.1%} | F1={m['f1']:.1%}")
    print("  Top 6 variables RF:", [(x["label"], x["importance"]) for x in fi_list[:6]])

    return rf_metrics


def train_nlp_4classes() -> dict:
    df = pd.read_csv(DATA_DIR / "radiology_reports_dataset.csv")
    X = df["texto_informe"].astype(str)
    y = df["categoria_rx"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=3500,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    C=2.2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    ordered_classes = ["Sin neumonía", "Posible", "Probable", "Definitiva"]

    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    vocab = np.array(vec.get_feature_names_out())
    clf_classes = list(clf.classes_)

    top_terms = {}
    for cls_name in ordered_classes:
        idx = clf_classes.index(cls_name)
        top_idx = np.argsort(clf.coef_[idx])[-8:][::-1]
        top_terms[cls_name] = [
            {"termino": str(vocab[i]), "peso": round(float(clf.coef_[idx][i]), 3)}
            for i in top_idx
        ]

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=ordered_classes)

    metrics = {
        "n_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro")), 4),
        "cv_f1_macro_mean": round(float(np.mean(cv_f1)), 4),
        "classes": ordered_classes,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c: {
                "precision": round(float(report_dict[c]["precision"]), 4),
                "recall": round(float(report_dict[c]["recall"]), 4),
                "f1": round(float(report_dict[c]["f1-score"]), 4),
            }
            for c in ordered_classes
        },
        "top_terms": top_terms,
    }

    joblib.dump(
        {"pipeline": pipeline, "classes": list(pipeline.classes_), "metrics": metrics},
        MODELS_DIR / "nlp_radiology_model.joblib",
    )

    print("\n=== NLP CLÍNICO 4 CATEGORÍAS (TEST SET) ===")
    print(f"  Accuracy={metrics['accuracy']:.1%} | F1 Macro={metrics['f1_macro']:.1%} | CV F1={metrics['cv_f1_macro_mean']:.1%}")
    print("  Matriz 4x4 (Sin neumonía, Posible, Probable, Definitiva):")
    for row_lbl, row_vals in zip(ordered_classes, cm.tolist()):
        print(f"    {row_lbl:14s}: {row_vals}")

    return metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rf_metrics = train_and_compare_clinical()
    nlp_metrics = train_nlp_4classes()

    summary = {
        "random_forest_clinico": rf_metrics,
        "nlp_informes_rx": nlp_metrics,
        "nota_academica": (
            "Dataset clínico sintético calibrado con el Puntaje de Heckerling (Ann Intern Med, 1990) "
            "y predictores de Moore et al. (Eur Respir J, 2017), estructurado para recolectar y "
            "reentrenar con fichas reales de urgencia chilena."
        ),
    }
    out_path = MODELS_DIR / "model_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMétricas guardadas en: {out_path}")


if __name__ == "__main__":
    main()
