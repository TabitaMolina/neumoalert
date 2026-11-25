"""
Motor de Fusión Multimodal de NEUMOALERT v3 (Heckerling + Random Forest + CNN + NLP 4 categorías).

Integra hasta 3 señales independientes:
1. Evaluación Clínica (Puntaje de Heckerling + Random Forest con variables adicionales del médico)
2. Radiografía de Tórax (Red Neuronal Convolucional - CNN)
3. Informe Médico / Radiológico PDF (NLP Clínico en 4 categorías: Sin neumonía, Posible, Probable, Definitiva)

Soporta dos momentos clínicos claros:
- Antes del informe radiológico (Triage): combina CNN + Evaluación Clínica (+ antecedente previo si existe)
- Con informe del radiólogo (Doble Chequeo): suma el NLP y alerta si hay discrepancia crítica
"""

from __future__ import annotations

from typing import Any


def evaluar_concordancia(
    prob_cnn: float | None,
    prob_rf: float | None,
    prob_nlp: float | None,
    es_antecedente_previo: bool = False,
) -> dict[str, str]:
    """Compara si las fuentes coinciden o si existe una discrepancia que requiera segunda revisión."""
    if prob_nlp is not None and not es_antecedente_previo and prob_cnn is not None:
        if prob_cnn >= 0.65 and prob_nlp < 0.30:
            return {
                "estado": "DISCREPANCIA CRÍTICA",
                "color": "rojo",
                "mensaje": (
                    "La radiografía (CNN) marca alta sospecha de neumonía, pero el informe escrito "
                    "la descarta. Se recomienda segunda lectura prioritaria para evitar un falso negativo."
                ),
            }
        if prob_nlp >= 0.65 and prob_cnn < 0.35:
            return {
                "estado": "DISCREPANCIA A REVISAR",
                "color": "amarillo",
                "mensaje": (
                    "El informe del radiólogo sugiere neumonía, pero la CNN marcó probabilidad baja. "
                    "Conviene revisar la calidad técnica o postura de la imagen."
                ),
            }
        if abs(prob_cnn - prob_nlp) <= 0.25:
            return {
                "estado": "ALTA CONCORDANCIA",
                "color": "verde",
                "mensaje": "La radiografía (CNN) y el texto del informe (NLP) coinciden.",
            }
        return {
            "estado": "CONCORDANCIA PARCIAL",
            "color": "amarillo",
            "mensaje": "Hay coincidencia moderada entre radiografía e informe; correlacionar con el examen físico.",
        }

    if prob_cnn is not None and prob_rf is not None:
        if abs(prob_cnn - prob_rf) <= 0.28:
            return {
                "estado": "ALTA CONCORDANCIA (CLÍNICA + RX)",
                "color": "verde",
                "mensaje": "La ficha clínica (Heckerling / Random Forest) y la radiografía apuntan en la misma dirección.",
            }
        return {
            "estado": "REVISAR CONTEXTO CLÍNICO",
            "color": "amarillo",
            "mensaje": (
                "Hay diferencia entre la sospecha clínica y la imagen. "
                "Recordar: una neumonía muy inicial puede no verse aún en la radiografía, o una sombra puede no ser neumonía."
            ),
        }

    return {
        "estado": "EVALUACIÓN CON 1 FUENTE",
        "color": "azul",
        "mensaje": "Estimación calculada solo con la ficha clínica (puedes sumar radiografía y/o informe PDF).",
    }


def fusionar_predicciones(
    datos_clinicos: dict[str, Any],
    prob_rf: float | None = None,
    prob_cnn: float | None = None,
    prob_nlp: float | None = None,
    modo: str = "Con informe",
    es_antecedente_previo: bool = False,
) -> dict[str, Any]:
    """Combina las probabilidades disponibles y entrega prioridad (ROJO/AMARILLO/VERDE) y recomendación clara."""
    es_modo_triage = "antes" in modo.lower() or "triage" in modo.lower()

    if es_modo_triage or es_antecedente_previo:
        base_weights = {"cnn": 0.60, "rf": 0.40, "nlp": 0.0}
    else:
        base_weights = {"cnn": 0.45, "rf": 0.25, "nlp": 0.30}

    signals = {
        "cnn": prob_cnn,
        "rf": prob_rf,
        "nlp": None if (es_modo_triage or es_antecedente_previo) else prob_nlp,
    }

    active_weights = {k: base_weights[k] for k, v in signals.items() if v is not None and base_weights[k] > 0}
    total_w = sum(active_weights.values())

    if total_w > 0:
        norm_weights = {k: w / total_w for k, v_w in active_weights.items() for w in [v_w]}
        prob_integrada = sum(norm_weights[k] * float(signals[k]) for k in norm_weights)
    else:
        norm_weights = {}
        prob_integrada = 0.0

    ajuste_antecedente = 0.0
    if es_antecedente_previo and prob_nlp is not None and prob_nlp >= 0.45:
        ajuste_antecedente = 0.04
        prob_integrada = min(0.99, prob_integrada + ajuste_antecedente)

    prob_integrada = round(float(prob_integrada), 4)
    concordancia = evaluar_concordancia(prob_cnn, prob_rf, prob_nlp, es_antecedente_previo)

    sat = float(datos_clinicos.get("sat_o2", 97.0))
    fr = int(datos_clinicos.get("frecuencia_respiratoria", 18))
    conciencia = bool(datos_clinicos.get("compromiso_conciencia", False))
    temp = float(datos_clinicos.get("temperatura", 36.8))
    fc = int(datos_clinicos.get("frecuencia_cardiaca", 84))
    edad = int(datos_clinicos.get("edad", 45))

    motivos = []
    if sat < 90:
        motivos.append(f"SatO₂ crítica ({sat:.0f}%)")
    elif sat < 95:
        motivos.append(f"SatO₂ bajo 95% ({sat:.0f}%)")
    if conciencia:
        motivos.append("Compromiso de conciencia")
    if fr >= 30:
        motivos.append(f"Taquipnea severa ({fr} rpm)")
    if prob_integrada >= 0.72:
        motivos.append(f"Probabilidad integrada alta ({prob_integrada:.0%})")
    elif prob_integrada >= 0.40:
        motivos.append(f"Sospecha intermedia ({prob_integrada:.0%})")
    if concordancia["estado"] == "DISCREPANCIA CRÍTICA":
        motivos.append("Discrepancia crítica entre radiografía e informe")

    if (
        sat < 90
        or (conciencia and prob_integrada >= 0.40)
        or (prob_integrada >= 0.75 and (sat < 94 or fr >= 26 or fc > 110))
        or (concordancia["estado"] == "DISCREPANCIA CRÍTICA" and (sat < 95 or temp > 37.8))
    ):
        prioridad = "ROJO"
        accion_sugerida = "Atención prioritaria inmediata: evaluación médica urgente y revisión prioritaria de radiografía."
    elif (
        prob_integrada >= 0.38
        or sat < 95
        or fr >= 24
        or temp > 37.8
        or edad >= 75
        or "DISCREPANCIA" in concordancia["estado"]
    ):
        prioridad = "AMARILLO"
        accion_sugerida = "Atención preferente: evaluar pronto en urgencia y correlacionar con radiografía."
    else:
        prioridad = "VERDE"
        accion_sugerida = "Riesgo bajo según triaje demostrativo: manejo estándar."

    if not motivos:
        motivos.append("Signos vitales estables y baja probabilidad combinada")

    return {
        "prob_integrada": prob_integrada,
        "prioridad": prioridad,
        "accion_sugerida": accion_sugerida,
        "motivo": " · ".join(motivos),
        "pesos_aplicados": {k: round(v, 3) for k, v in norm_weights.items()},
        "ajuste_antecedente": ajuste_antecedente,
        "concordancia": concordancia,
        "modo_aplicado": "Antes del informe (Triage)" if (es_modo_triage or es_antecedente_previo) else "Con informe (Doble Chequeo)",
    }
