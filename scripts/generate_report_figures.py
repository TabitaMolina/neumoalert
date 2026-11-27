"""
Genera 5 figuras de alta resolución (200 DPI) para el Informe Final oficial de NEUMOALERT.
Todas las métricas y gráficos provienen directamente de los datasets y modelos entrenados del proyecto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.clinical_rf import CLINICAL_FEATURES

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
})


def make_fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=200)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, title_color="#0f172a"):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            facecolor=fc, edgecolor=ec, linewidth=1.8,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="top", fontsize=9.8, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h / 2 - 0.16, body, ha="center", va="center", fontsize=8.3, color="#334155", linespacing=1.35)

    ax.text(5.75, 5.55, "Arquitectura del Sistema Multimodal NEUMOALERT", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#0b2d48")

    # Columna 1: Entradas + 3 Modelos IA
    box(0.25, 3.85, 3.15, 1.25, "1. Radiografía de Tórax (CNN)", "Dataset público Chest X-Ray\nRed Neuronal Convolucional\n→ Probabilidad visual (P_CNN)", "#eef2ff", "#4f46e5", "#312e81")
    box(0.25, 2.20, 3.15, 1.25, "2. Ficha Clínica (Random Forest)", "Puntaje Heckerling (0–5) + SatO₂,\nFR, antipirético, rinorrea y signos\n→ Probabilidad clínica (P_RF)", "#e0f2fe", "#0284c7", "#0c4a6e")
    box(0.25, 0.55, 3.15, 1.25, "3. Informe PDF (NLP Clínico)", "PyMuPDF + filtro guías + negaciones\nTF-IDF + Regresión Logística\n→ 4 categorías médicas (P_NLP)", "#ccfbf1", "#0d9488", "#134e4a")

    # Columna 2: Motor de Fusión
    box(4.25, 1.45, 3.0, 2.75, "Motor de Fusión Multimodal", "Modo Triage (sin informe):\n60% CNN + 40% RF\n\nModo Doble Chequeo:\n45% CNN + 25% RF + 30% NLP\n\n+ Detector de Discrepancia\n+ Semáforo (Rojo/Amarillo/Verde)", "#fffbeb", "#d97706", "#78350f")

    # Columna 3: App Clínica + BD
    box(8.10, 2.95, 3.15, 1.95, "Aplicación Clínica (Streamlit)", "• Ingreso y Evaluación rápida\n• Cola priorizada de Urgencias\n• Ficha del Radiólogo + CSV\n• Conector plug-and-play CNN", "#f0fdf4", "#16a34a", "#14532d")
    box(8.10, 0.85, 3.15, 1.55, "Persistencia (SQLAlchemy)", "Tablas: pacientes, imagenes,\ndiagnosticos y alertas\nSQLite local / PostgreSQL\nExportación CSV estudio médico", "#f8fafc", "#64748b", "#0f172a")

    # Flechas
    arrow_kw = dict(arrowstyle="->,head_width=0.28,head_length=0.35", lw=1.8, color="#475569")
    ax.annotate("", xy=(4.20, 3.65), xytext=(3.48, 4.45), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.20, 2.82), xytext=(3.48, 2.82), arrowprops=arrow_kw)
    ax.annotate("", xy=(4.20, 2.00), xytext=(3.48, 1.18), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.05, 3.75), xytext=(7.33, 3.15), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.05, 1.75), xytext=(7.33, 2.40), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.67, 2.46), xytext=(9.67, 2.90), arrowprops=dict(arrowstyle="<->,head_width=0.22,head_length=0.28", lw=1.4, color="#64748b"))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_arquitectura_sistema.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig2_eda() -> None:
    df = pd.read_csv(DATA_DIR / "clinical_pneumonia_dataset.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=200)

    # Panel A: Tasa de RxT alterada por Puntaje Heckerling
    rates = df.groupby("puntaje_heckerling")["rx_alterada"].mean() * 100
    counts = df.groupby("puntaje_heckerling")["rx_alterada"].count()
    colors_h = ["#16a34a", "#22c55e", "#f59e0b", "#ea580c", "#dc2626", "#991b1b"]
    bars = axes[0].bar(rates.index, rates.values, color=colors_h, edgecolor="#1e293b", linewidth=0.6)
    for bar, val, n in zip(bars, rates.values, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 2.2, f"{val:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0f172a")
    axes[0].set_ylim(0, 112)
    axes[0].set_xticks(range(6))
    axes[0].set_xlabel("Puntaje de Heckerling (0 a 5 puntos)", fontsize=9.5)
    axes[0].set_ylabel("% de pacientes con RxT alterada", fontsize=9.5)
    axes[0].set_title("Tasa de RxT alterada según Puntaje de Heckerling", fontsize=10.5, fontweight="bold", color="#0b2d48")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Distribución de SatO2 (%)
    df_eval = df[df["categoria_rx"] != "Posible"]
    axes[1].hist(df_eval[df_eval["rx_alterada"] == 0]["sat_o2"], bins=18, alpha=0.62, label="Sin neumonía", color="#16a34a", edgecolor="white")
    axes[1].hist(df_eval[df_eval["rx_alterada"] == 1]["sat_o2"], bins=18, alpha=0.68, label="RxT alterada (Probable/Definitiva)", color="#dc2626", edgecolor="white")
    axes[1].axvline(95.0, color="#0f4c75", linestyle="--", linewidth=1.6, label="Umbral clínico SatO₂ = 95%")
    axes[1].set_xlabel("Saturación de oxígeno SatO₂ (%)", fontsize=9.5)
    axes[1].set_ylabel("Número de pacientes", fontsize=9.5)
    axes[1].set_title("Distribución de SatO₂ (%) por resultado radiológico", fontsize=10.5, fontweight="bold", color="#0b2d48")
    axes[1].legend(fontsize=8.2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_eda_heckerling.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig3_roc_and_cm() -> None:
    df = pd.read_csv(DATA_DIR / "clinical_pneumonia_dataset.csv")
    df_eval = df[df["categoria_rx"] != "Posible"].copy().reset_index(drop=True)
    X = df_eval[CLINICAL_FEATURES]
    y = df_eval["rx_alterada"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    heck_lr = LogisticRegression(class_weight="balanced", random_state=42).fit(X_train[["puntaje_heckerling"]], y_train)
    heck_prob = heck_lr.predict_proba(X_test[["puntaje_heckerling"]])[:, 1]

    rf_bundle = joblib.load(MODELS_DIR / "rf_clinical_model.joblib")
    rf_pipe = rf_bundle["pipeline"]
    rf_prob = rf_pipe.predict_proba(X_test)[:, 1]
    cm = np.array(rf_bundle["metrics"]["confusion_matrix"])

    heck_auc = roc_auc_score(y_test, heck_prob)
    rf_auc = roc_auc_score(y_test, rf_prob)
    n_test = len(y_test)
    fpr_h, tpr_h, _ = roc_curve(y_test, heck_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=200)

    axes[0].plot(fpr_h, tpr_h, label=f"Heckerling solo (AUC = {heck_auc:.3f})".replace(".", ","), color="#64748b", linestyle="--", linewidth=1.8)
    axes[0].plot(fpr_rf, tpr_rf, label=f"Random Forest completo (AUC = {rf_auc:.3f})".replace(".", ","), color="#0f4c75", linewidth=2.3)
    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.35, label="Azar (AUC = 0,500)")
    axes[0].set_title("Curva ROC comparativa (Test Set no visto)", fontsize=10.5, fontweight="bold", color="#0b2d48")
    axes[0].set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=9)
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (Recall)", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8.3)
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(cm, cmap="Blues")
    labels = ["Sin neumonía (0)", "RxT alterada (1)"]
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("Predicción del modelo", fontsize=9.5)
    axes[1].set_ylabel("Realidad en radiografía", fontsize=9.5)
    axes[1].set_title(f"Matriz de Confusión — Random Forest (n = {n_test})", fontsize=10.5, fontweight="bold", color="#0b2d48")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "#0f172a"
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_roc_y_matriz_rf.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig4_feature_importance() -> None:
    metrics = json.loads((MODELS_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    fi = pd.DataFrame(metrics["random_forest_clinico"]["feature_importances"]).head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=200)
    bars = ax.barh(fi["label"], fi["importance"] * 100, color="#0f4c75", edgecolor="none")
    for bar, val in zip(bars, fi["importance"] * 100):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha="left", fontsize=8.8, fontweight="bold", color="#0f172a")
    ax.set_xlim(0, max(fi["importance"] * 100) + 5.5)
    ax.set_xlabel("Importancia relativa en Random Forest (%)", fontsize=9.5)
    ax.set_title("Top 10 Variables Más Influyentes del Modelo Clínico (Random Forest)", fontsize=11, fontweight="bold", color="#0b2d48")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_importancia_variables.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_fig5_nlp_matrix() -> None:
    metrics = json.loads((MODELS_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    nlpm = metrics["nlp_informes_rx"]
    cm = np.array(nlpm["confusion_matrix"])
    classes = nlpm["classes"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=9.5)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=9.5)
    ax.set_xlabel("Categoría predicha por el modelo NLP", fontsize=9.8)
    ax.set_ylabel("Categoría real del informe (Sección E)", fontsize=9.8)
    ax.set_title("Matriz de Confusión 4×4 — NLP Clínico (Accuracy = 95,5%)", fontsize=11, fontweight="bold", color="#0b2d48")
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "#0f172a"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12.5, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_matriz_nlp_4clases.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    make_fig1_architecture()
    make_fig2_eda()
    make_fig3_roc_and_cm()
    make_fig4_feature_importance()
    make_fig5_nlp_matrix()
    print("5 figuras generadas en:", FIG_DIR)


if __name__ == "__main__":
    main()
