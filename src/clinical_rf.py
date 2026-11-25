"""
Módulo clínico de NEUMOALERT alineado a la ficha del médico (Puntaje de Heckerling + Datos adicionales).

Incluye:
- Cálculo exacto del Puntaje de Heckerling (0 a 5) y su tabla de referencia publicada (Heckerling, 1990).
- Inferencia con el modelo Random Forest entrenado con Heckerling + variables modernas del médico
  (SatO2, FR, antipirético previo, rinorrea/odinofagia, matidez, egofonía, soplo tubárico,
  compromiso de conciencia, enfermedades previas, edad, sexo y días de evolución).
- Explicación en español simple de los factores que más pesaron en el paciente evaluado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RF_MODEL_PATH = BASE_DIR / "models" / "rf_clinical_model.joblib"

CLINICAL_FEATURES = [
    # Sección B: Enfermedades previas
    "asma",
    "epoc",
    "otra_enf_pulmonar",
    "insuficiencia_cardiaca",
    "inmunosupresion",
    "demencia",
    # Sección C: Criterios de Heckerling
    "temperatura",
    "frecuencia_cardiaca",
    "crepitantes",
    "murmullo_disminuido",
    "ausencia_asma",
    "puntaje_heckerling",
    # Sección D: Datos adicionales del médico
    "frecuencia_respiratoria",
    "sat_o2",
    "matidez",
    "egofonia",
    "soplo_tubarico",
    "compromiso_conciencia",
    "rinorrea_odinofagia",
    "antipiretico_6h",
    "dias_evolucion",
    "edad",
    "sexo",
]

FEATURE_LABELS_ES = {
    "puntaje_heckerling": "Puntaje de Heckerling (0 a 5)",
    "sat_o2": "Saturación de oxígeno (SatO₂ %)",
    "temperatura": "Temperatura axilar (°C)",
    "frecuencia_cardiaca": "Frecuencia cardíaca (FC lpm)",
    "frecuencia_respiratoria": "Frecuencia respiratoria (FR rpm)",
    "crepitantes": "Crepitaciones en auscultación",
    "murmullo_disminuido": "Murmullo disminuido localizado",
    "ausencia_asma": "Ausencia de asma",
    "asma": "Antecedente de asma",
    "rinorrea_odinofagia": "Rinorrea / dolor de garganta (sugiere resfrío)",
    "antipiretico_6h": "Antipirético últimas 6 h (puede ocultar fiebre)",
    "compromiso_conciencia": "Compromiso de conciencia",
    "matidez": "Matidez a la percusión",
    "egofonia": "Egofonía",
    "soplo_tubarico": "Soplo tubárico",
    "edad": "Edad (años)",
    "dias_evolucion": "Días de evolución",
    "epoc": "EPOC",
    "insuficiencia_cardiaca": "Insuficiencia cardíaca (ICC)",
    "inmunosupresion": "Inmunosupresión",
    "otra_enf_pulmonar": "Otra enfermedad pulmonar",
    "demencia": "Demencia",
    "sexo": "Sexo",
}

# Tabla publicada de Heckerling et al., Ann Intern Med 1990 (n = 1.134 pacientes en urgencia)
HECKERLING_REF_TABLE = {
    0: {"prob_ref": 0.020, "zona": "Riesgo muy bajo", "detalle": "Casi siempre la radiografía sale normal (~2% alterada)."},
    1: {"prob_ref": 0.034, "zona": "Riesgo bajo", "detalle": "Casi siempre la radiografía sale normal (~3,4% alterada)."},
    2: {"prob_ref": 0.108, "zona": "Zona gris", "detalle": "Zona intermedia (~11% en Chicago): el puntaje solo ayuda poco a decidir."},
    3: {"prob_ref": 0.220, "zona": "Zona gris alta", "detalle": "Sospecha intermedia (~22% en Chicago): conviene apoyar con SatO₂ e IA."},
    4: {"prob_ref": 0.552, "zona": "Alta sospecha", "detalle": "Más de la mitad tiene radiografía alterada (~55% en Chicago)."},
    5: {"prob_ref": 0.750, "zona": "Muy alta sospecha", "detalle": "3 de cada 4 pacientes presentan infiltrado en la radiografía (~75%)."},
}

_RF_CACHE: dict[str, Any] | None = None


def reload_rf_cache() -> None:
    global _RF_CACHE
    _RF_CACHE = None


def load_rf_bundle() -> dict[str, Any]:
    global _RF_CACHE
    if _RF_CACHE is None:
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo Random Forest en {RF_MODEL_PATH}. "
                "Ejecuta primero `python scripts/train_models.py`."
            )
        _RF_CACHE = joblib.load(RF_MODEL_PATH)
    return _RF_CACHE


def calcular_heckerling(datos: dict[str, Any]) -> dict[str, Any]:
    """Calcula el Puntaje de Heckerling exacto (0 a 5) y desglosa cada criterio."""
    temp = float(datos.get("temperatura", 36.8))
    fc = int(datos.get("frecuencia_cardiaca", 84))
    crep = bool(datos.get("crepitantes", False))
    murm = bool(datos.get("murmullo_disminuido", False))
    asma = bool(datos.get("asma", False))

    criterios = [
        ("Temperatura > 37,8 °C", temp > 37.8, f"{temp:.1f} °C"),
        ("FC > 100 lpm", fc > 100, f"{fc} lpm"),
        ("Crepitaciones", crep, "Sí" if crep else "No"),
        ("Murmullo disminuido localizado", murm, "Sí" if murm else "No"),
        ("Ausencia de asma", not asma, "Sin asma (+1)" if not asma else "Tiene asma (0)"),
    ]
    puntaje = sum(1 for _, activo, _ in criterios if activo)
    ref = HECKERLING_REF_TABLE[puntaje]
    return {
        "puntaje": puntaje,
        "criterios": criterios,
        "prob_referencia_1990": ref["prob_ref"],
        "zona": ref["zona"],
        "detalle": ref["detalle"],
    }


def explicar_factores_paciente(datos: dict[str, Any], heck: dict[str, Any]) -> dict[str, list[str]]:
    """Separa factores a favor y en contra de neumonía en lenguaje claro para el clínico."""
    a_favor = []
    en_contra = []
    notas = []

    temp = float(datos.get("temperatura", 36.8))
    fc = int(datos.get("frecuencia_cardiaca", 84))
    sat = float(datos.get("sat_o2", 97.0))
    fr = int(datos.get("frecuencia_respiratoria", 18))

    if temp > 37.8:
        a_favor.append(f"Fiebre axilar ({temp:.1f} °C > 37,8 °C)")
    if fc > 100:
        a_favor.append(f"Pulso acelerado (FC {fc} lpm > 100)")
    if bool(datos.get("crepitantes")):
        a_favor.append("Crepitaciones en auscultación")
    if bool(datos.get("murmullo_disminuido")):
        a_favor.append("Murmullo pulmonar disminuido localizado")
    if sat < 90:
        a_favor.append(f"Hipoxemia crítica (SatO₂ {sat:.0f}% < 90%)")
    elif sat < 95:
        a_favor.append(f"Saturación de oxígeno baja (SatO₂ {sat:.0f}% < 95%)")
    if fr >= 24:
        a_favor.append(f"Respiración agitada (FR {fr} rpm)")
    if bool(datos.get("compromiso_conciencia")):
        a_favor.append("Compromiso de conciencia / desorientación nueva")
    for campo, texto in [
        ("matidez", "Matidez a la percusión"),
        ("egofonia", "Egofonía presente"),
        ("soplo_tubarico", "Soplo tubárico presente"),
    ]:
        if bool(datos.get(campo)):
            a_favor.append(texto)

    if bool(datos.get("rinorrea_odinofagia")):
        en_contra.append("Rinorrea / dolor de garganta (apunta más a resfrío viral)")
    if bool(datos.get("asma")):
        en_contra.append("Tiene asma (puede explicar ruidos pulmonares y pulso alto sin neumonía)")
    if sat >= 97 and temp <= 37.5 and not bool(datos.get("crepitantes")):
        en_contra.append("Signos vitales estables y sin crepitaciones")

    if bool(datos.get("antipiretico_6h")) and temp <= 37.8:
        notas.append("⚠️ Tomó antipirético en las últimas 6 h: la fiebre real podría estar oculta y restar 1 punto en Heckerling.")
    if int(datos.get("dias_evolucion", 3)) <= 1:
        notas.append("ℹ️ Cuadro muy reciente (≤ 1 día): una neumonía inicial a veces aún no es visible en la radiografía.")

    return {"a_favor": a_favor, "en_contra": en_contra, "notas": notas}


def predict_clinical_rf(datos: dict[str, Any]) -> dict[str, Any]:
    """Calcula Puntaje de Heckerling + probabilidad con Random Forest ampliado."""
    bundle = load_rf_bundle()
    pipeline = bundle["pipeline"]
    feature_importances = bundle.get("feature_importances", [])

    asma = int(bool(datos.get("asma", False)))
    temp = float(datos.get("temperatura", 36.8))
    fc = int(datos.get("frecuencia_cardiaca", 84))
    crep = int(bool(datos.get("crepitantes", False)))
    murm = int(bool(datos.get("murmullo_disminuido", False)))
    ausencia_asma = 1 - asma

    heck = calcular_heckerling(
        {
            "temperatura": temp,
            "frecuencia_cardiaca": fc,
            "crepitantes": crep,
            "murmullo_disminuido": murm,
            "asma": asma,
        }
    )

    sexo_raw = str(datos.get("sexo", "F"))
    sexo_norm = "M" if sexo_raw.strip().upper().startswith("M") else "F"

    row = {
        "asma": asma,
        "epoc": int(bool(datos.get("epoc", False))),
        "otra_enf_pulmonar": int(bool(datos.get("otra_enf_pulmonar", False))),
        "insuficiencia_cardiaca": int(bool(datos.get("insuficiencia_cardiaca", False))),
        "inmunosupresion": int(bool(datos.get("inmunosupresion", False))),
        "demencia": int(bool(datos.get("demencia", False))),
        "temperatura": temp,
        "frecuencia_cardiaca": fc,
        "crepitantes": crep,
        "murmullo_disminuido": murm,
        "ausencia_asma": ausencia_asma,
        "puntaje_heckerling": heck["puntaje"],
        "frecuencia_respiratoria": int(datos.get("frecuencia_respiratoria", 18)),
        "sat_o2": float(datos.get("sat_o2", 97.0)),
        "matidez": int(bool(datos.get("matidez", False))),
        "egofonia": int(bool(datos.get("egofonia", False))),
        "soplo_tubarico": int(bool(datos.get("soplo_tubarico", False))),
        "compromiso_conciencia": int(bool(datos.get("compromiso_conciencia", False))),
        "rinorrea_odinofagia": int(bool(datos.get("rinorrea_odinofagia", False))),
        "antipiretico_6h": int(bool(datos.get("antipiretico_6h", False))),
        "dias_evolucion": int(datos.get("dias_evolucion", 4)),
        "edad": int(datos.get("edad", 45)),
        "sexo": sexo_norm,
    }

    df_in = pd.DataFrame([row], columns=CLINICAL_FEATURES)
    prob_rf = float(pipeline.predict_proba(df_in)[0][1])

    if prob_rf >= 0.68:
        categoria = "Alta sospecha de radiografía alterada"
    elif prob_rf >= 0.35:
        categoria = "Sospecha intermedia (zona gris)"
    else:
        categoria = "Baja sospecha clínica"

    explicacion = explicar_factores_paciente(row, heck)

    return {
        "prob_rf": round(prob_rf, 4),
        "categoria_rf": categoria,
        "heckerling": heck,
        "explicacion": explicacion,
        "feature_importances": feature_importances,
        "features_usadas": row,
    }
