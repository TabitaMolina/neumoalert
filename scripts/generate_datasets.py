"""
PASO 1 · Generador de datasets clínicos alineados a la ficha del médico (Puntaje de Heckerling).

Estructura basada exactamente en el cuestionario médico del proyecto:
- Sección A: Criterios de ingreso (adultos >= 18 años, síntomas < 14 días, indicación de RxT)
- Sección B: Enfermedades previas (asma, EPOC, otra pulmonar, ICC, inmunosupresión, demencia)
- Sección C: Puntaje de Heckerling (0 a 5 puntos):
    1) Temperatura axilar > 37,8 °C
    2) FC > 100 lpm
    3) Crepitaciones
    4) Murmullo disminuido localizado
    5) Ausencia de asma
- Sección D: Datos adicionales (SatO2, FR, matidez, egofonía, soplo tubárico,
    compromiso de conciencia, rinorrea/odinofagia, antipirético en últimas 6h, días, edad, sexo)
- Sección E: Resultado radiológico en 4 categorías exactas de Heckerling:
    1) Sin neumonía
    2) Posible
    3) Probable
    4) Definitiva

NOTA DE TRANSPARENCIA ACADÉMICA:
Datos sintéticos calibrados con las probabilidades publicadas de Heckerling (1990)
y Moore et al. (2017) con fines demostrativos hasta contar con fichas reales de urgencia.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "sample_pdfs"


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def calcular_heckerling_fila(
    temperatura: float,
    frecuencia_cardiaca: int,
    crepitantes: int,
    murmullo_disminuido: int,
    asma: int,
) -> int:
    """Calcula el Puntaje de Heckerling exacto (0 a 5 puntos)."""
    return (
        int(temperatura > 37.8)
        + int(frecuencia_cardiaca > 100)
        + int(bool(crepitantes))
        + int(bool(murmullo_disminuido))
        + int(not bool(asma))
    )


def generate_heckerling_clinical_dataset(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Genera 2.500 adultos (>= 18 años) que cumplen los criterios de ingreso (Sección A),
    calibrados para reproducir el comportamiento real publicado del Puntaje de Heckerling
    (AUC ~0,82) y el valor agregado de las variables modernas de la Sección D (SatO2,
    antipirético previo, rinorrea/odinofagia, signos focales y compromiso de conciencia).
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Perfiles realistas de consulta de urgencia adulta con indicación de RxT
    # (prevalencia moderada-realista ~28% de RxT alterada Probable/Definitiva)
    profiles = rng.choice(
        ["resfrio_viral", "bronquitis", "asma_crisis", "epoc_icc", "neumonia_leve", "neumonia_franca"],
        size=n_samples,
        p=[0.24, 0.22, 0.14, 0.12, 0.14, 0.14],
    )

    for idx, prof in enumerate(profiles, start=1):
        sexo = str(rng.choice(["F", "M"], p=[0.52, 0.48]))

        # B. Enfermedades previas
        asma = int(prof == "asma_crisis" or (prof != "asma_crisis" and rng.random() < 0.08))
        epoc = int((prof == "epoc_icc" and rng.random() < 0.65) or (prof != "epoc_icc" and rng.random() < 0.09))
        otra_pulmonar = int(rng.random() < 0.05)
        icc = int((prof == "epoc_icc" and rng.random() < 0.45) or (prof != "epoc_icc" and rng.random() < 0.07))
        inmunosupresion = int(rng.random() < 0.08)

        if prof == "resfrio_viral":
            edad = int(np.clip(rng.normal(38, 15), 18, 86))
            temp_real = float(np.clip(rng.normal(37.2, 0.55), 35.9, 39.0))
            fc = int(np.clip(rng.normal(86, 12), 58, 122))
            crepitantes = int(rng.random() < 0.06)
            murmullo_dism = int(rng.random() < 0.04)
            fr = int(np.clip(rng.normal(17, 2.5), 12, 26))
            sat_o2 = round(float(np.clip(rng.normal(97.8, 1.1), 94.0, 100.0)), 1)
            rinorrea_odinof = int(rng.random() < 0.78)
            matidez, egofonia, soplo = 0, 0, 0
            conciencia = int(rng.random() < 0.01)
            antipiretico = int(rng.random() < 0.35)
            dias = int(np.clip(rng.normal(3.5, 2.0), 1, 13))
            riesgo_latente = -2.4

        elif prof == "bronquitis":
            edad = int(np.clip(rng.normal(46, 16), 18, 88))
            temp_real = float(np.clip(rng.normal(37.4, 0.55), 36.0, 39.1))
            fc = int(np.clip(rng.normal(90, 13), 60, 125))
            crepitantes = int(rng.random() < 0.14)
            murmullo_dism = int(rng.random() < 0.08)
            fr = int(np.clip(rng.normal(19, 3.0), 13, 28))
            sat_o2 = round(float(np.clip(rng.normal(96.8, 1.5), 92.0, 100.0)), 1)
            rinorrea_odinof = int(rng.random() < 0.42)
            matidez = int(rng.random() < 0.02)
            egofonia = int(rng.random() < 0.01)
            soplo = int(rng.random() < 0.01)
            conciencia = int(rng.random() < 0.02)
            antipiretico = int(rng.random() < 0.30)
            dias = int(np.clip(rng.normal(6.5, 3.0), 1, 13))
            riesgo_latente = -1.9

        elif prof == "asma_crisis":
            edad = int(np.clip(rng.normal(36, 15), 18, 80))
            temp_real = float(np.clip(rng.normal(36.9, 0.45), 35.9, 38.4))
            fc = int(np.clip(rng.normal(106, 12), 75, 138))
            crepitantes = int(rng.random() < 0.18)
            murmullo_dism = int(rng.random() < 0.20)
            fr = int(np.clip(rng.normal(24, 3.5), 16, 36))
            sat_o2 = round(float(np.clip(rng.normal(94.8, 2.0), 89.0, 99.0)), 1)
            rinorrea_odinof = int(rng.random() < 0.32)
            matidez, egofonia, soplo = 0, 0, 0
            conciencia = int(rng.random() < 0.02)
            antipiretico = int(rng.random() < 0.15)
            dias = int(np.clip(rng.normal(3.0, 2.0), 1, 12))
            riesgo_latente = -2.3

        elif prof == "epoc_icc":
            edad = int(np.clip(rng.normal(71, 10), 45, 93))
            temp_real = float(np.clip(rng.normal(37.1, 0.55), 35.8, 38.8))
            fc = int(np.clip(rng.normal(97, 14), 65, 135))
            crepitantes = int(rng.random() < 0.45)
            murmullo_dism = int(rng.random() < 0.25)
            fr = int(np.clip(rng.normal(23, 4.0), 15, 35))
            sat_o2 = round(float(np.clip(rng.normal(93.6, 2.4), 86.0, 99.0)), 1)
            rinorrea_odinof = int(rng.random() < 0.20)
            matidez = int(rng.random() < 0.05)
            egofonia = int(rng.random() < 0.02)
            soplo = int(rng.random() < 0.02)
            conciencia = int(rng.random() < 0.08)
            antipiretico = int(rng.random() < 0.22)
            dias = int(np.clip(rng.normal(5.0, 3.0), 1, 13))
            riesgo_latente = -1.2

        elif prof == "neumonia_leve":
            edad = int(np.clip(rng.normal(56, 19), 18, 92))
            temp_real = float(np.clip(rng.normal(38.2, 0.65), 36.2, 40.0))
            fc = int(np.clip(rng.normal(101, 13), 70, 136))
            crepitantes = int(rng.random() < 0.58)
            murmullo_dism = int(rng.random() < 0.42)
            fr = int(np.clip(rng.normal(23, 4.0), 15, 36))
            sat_o2 = round(float(np.clip(rng.normal(93.8, 2.3), 86.0, 98.5)), 1)
            rinorrea_odinof = int(rng.random() < 0.16)
            matidez = int(rng.random() < 0.14)
            egofonia = int(rng.random() < 0.09)
            soplo = int(rng.random() < 0.08)
            conciencia = int(rng.random() < (0.22 if edad >= 65 else 0.04))
            antipiretico = int(rng.random() < 0.45)
            dias = int(np.clip(rng.normal(5.5, 2.8), 1, 13))
            riesgo_latente = 0.85

        else:  # neumonia_franca
            edad = int(np.clip(rng.normal(64, 17), 18, 94))
            temp_real = float(np.clip(rng.normal(38.8, 0.60), 36.8, 40.5))
            fc = int(np.clip(rng.normal(111, 12), 78, 145))
            crepitantes = int(rng.random() < 0.82)
            murmullo_dism = int(rng.random() < 0.68)
            fr = int(np.clip(rng.normal(27, 4.5), 18, 42))
            sat_o2 = round(float(np.clip(rng.normal(90.8, 2.9), 82.0, 97.5)), 1)
            rinorrea_odinof = int(rng.random() < 0.08)
            matidez = int(rng.random() < 0.28)
            egofonia = int(rng.random() < 0.18)
            soplo = int(rng.random() < 0.16)
            conciencia = int(rng.random() < (0.34 if edad >= 65 else 0.07))
            antipiretico = int(rng.random() < 0.42)
            dias = int(np.clip(rng.normal(5.0, 2.5), 1, 13))
            riesgo_latente = 2.15

        demencia = int(edad >= 70 and rng.random() < 0.14)

        # Efecto realista señalado por el médico:
        # Si el paciente tomó antipirético en las últimas 6 h, la temperatura axilar medida baja ~0,7-1,1 °C
        # y puede "engañar" al puntaje de Heckerling quitándole el punto de fiebre!
        if antipiretico == 1 and temp_real > 37.6:
            temp_medida = round(float(np.clip(temp_real - rng.uniform(0.65, 1.15), 35.8, 40.3)), 1)
        else:
            temp_medida = round(temp_real, 1)

        ausencia_asma = 1 - asma
        heckerling = calcular_heckerling_fila(temp_medida, fc, crepitantes, murmullo_dism, asma)

        # Probabilidad clínica calibrada a la tabla publicada de Heckerling (1990)
        # + efecto real de las variables modernas de la Sección D (SatO2, antipirético, rinorrea, signos focales)
        logit = (
            0.45 * riesgo_latente
            + 0.70 * int(temp_real > 37.8)  # fiebre real subyacente (incluso enmascarada por paracetamol!)
            + 0.48 * int(fc > 100)
            + 0.72 * crepitantes
            + 0.68 * murmullo_dism
            + 0.52 * ausencia_asma
            - 0.36 * (sat_o2 - 95.0)        # SatO2 < 95% (Moore et al. 2017)
            + 0.08 * (fr - 20.0)
            - 1.05 * rinorrea_odinof        # resfrío apunta EN CONTRA de neumonía
            + 1.05 * matidez
            + 0.95 * egofonia
            + 1.00 * soplo
            + 0.90 * conciencia             # 3x más frecuente en neumonía según Heckerling
            + 0.25 * inmunosupresion
            - 0.25 * (1 if dias <= 1 else 0)
            + rng.normal(0, 1.05)
        )
        prob_rx = sigmoid(logit - 2.35)

        # Categoría radiológica en 4 niveles exactos de la Sección E del médico
        u = rng.random()
        if u < prob_rx:
            categoria_rx = "Definitiva" if (prob_rx > 0.62 and rng.random() < 0.70) or rng.random() < 0.50 else "Probable"
            rx_alterada = 1
        else:
            if 0.15 <= prob_rx <= 0.65 and rng.random() < 0.28:
                categoria_rx = "Posible"
            else:
                categoria_rx = "Sin neumonía"
            rx_alterada = 0

        rows.append(
            {
                "paciente_num": f"PAC-{idx:04d}",
                # Sección B: Enfermedades previas
                "asma": asma,
                "epoc": epoc,
                "otra_enf_pulmonar": otra_pulmonar,
                "insuficiencia_cardiaca": icc,
                "inmunosupresion": inmunosupresion,
                "demencia": demencia,
                # Sección C: Criterios Heckerling
                "temperatura": temp_medida,
                "frecuencia_cardiaca": fc,
                "crepitantes": crepitantes,
                "murmullo_disminuido": murmullo_dism,
                "ausencia_asma": ausencia_asma,
                "puntaje_heckerling": heckerling,
                # Sección D: Datos adicionales del médico
                "frecuencia_respiratoria": fr,
                "sat_o2": sat_o2,
                "matidez": matidez,
                "egofonia": egofonia,
                "soplo_tubarico": soplo,
                "compromiso_conciencia": conciencia,
                "rinorrea_odinofagia": rinorrea_odinof,
                "antipiretico_6h": antipiretico,
                "dias_evolucion": dias,
                "edad": edad,
                "sexo": sexo,
                # Sección E: Desenlace radiológico (4 categorías exactas)
                "categoria_rx": categoria_rx,
                "rx_alterada": rx_alterada,  # 1 = Probable o Definitiva (como en Heckerling 1990)
            }
        )

    return pd.DataFrame(rows)


def generate_radiology_reports_4classes(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Genera 1.000 informes radiológicos en español etiquetados exactamente en las
    4 categorías de la Sección E del estudio del médico:
      - Sin neumonía
      - Posible
      - Probable
      - Definitiva
    """
    rng = random.Random(seed)

    headers = [
        "INFORME RADIOLÓGICO - RADIOGRAFÍA DE TÓRAX PA Y LATERAL.",
        "INFORME RADIOLÓGICO DE TÓRAX PORTÁTIL.",
        "ESTUDIO: RADIOGRAFÍA DE TÓRAX AP Y LATERAL.",
        "RADIOGRAFÍA DE TÓRAX FRONTAL Y LATERAL. TÉCNICA ADECUADA.",
    ]
    motivos = [
        "Motivo de estudio: fiebre y tos, descartar neumonía.",
        "Motivo: sospecha clínica de neumonía en urgencia.",
        "Indicación: disnea y crepitaciones, evaluar infiltrados pulmonares.",
        "Antecedente clínico: cuadro respiratorio agudo < 14 días.",
    ]
    regiones = [
        "lóbulo inferior derecho",
        "base pulmonar izquierda",
        "lóbulo medio derecho",
        "campo pulmonar inferior izquierdo",
        "región parahiliar derecha",
        "ambas bases pulmonares",
    ]
    normales_extra = [
        "Silueta cardiomediastínica de tamaño y morfología normal.",
        "Senos costofrénicos libres y agudos.",
        "Estructuras óseas del tórax sin lesiones agudas.",
        "Tráquea central, mediastino de amplitud conservada.",
    ]

    # 1. Definitiva: infiltrado neumónico inequívoco
    def_hall = [
        "Extensa consolidación parenquimatosa homogénea en {reg} con broncograma aéreo evidente en su interior.",
        "Foco de condensación alveolar denso y bien definido en {reg}, acompañado de derrame pleural ipsilateral.",
        "Opacidad alveolar confluente inequívoca en {reg} con borramiento del contorno diafragmático.",
        "Infiltrado neumónico lobar franco en {reg} con signos claros de relleno alveolar agudo.",
    ]
    def_conc = [
        "Impresión diagnóstica: Neumonía confirmada radiográficamente en {reg}. Infiltrado neumónico inequívoco.",
        "Conclusión radiológica: Consolidación neumónica definitiva en {reg}.",
        "Impresión diagnóstica: Cuadro radiográfico categórico de neumonía adquirida en la comunidad en {reg}.",
        "Conclusión: Evidencia clara e inequívoca de neumonía basal en {reg}.",
    ]

    # 2. Probable: "probablemente" o "lo más probable" es neumonía
    prob_hall = [
        "Opacidad alveolar parcheada en {reg}, de densidad moderada, sin derrame pleural cuantioso.",
        "Infiltrado parenquimatoso focal en {reg} con discreto refuerzo peribronquial adyacente.",
        "Aumento de densidad heterogénea en {reg} con contornos parcialmente definidos.",
        "Foco de opacidad en {reg} con leve borramiento vascular local.",
    ]
    prob_conc = [
        "Impresión diagnóstica: Hallazgos que probablemente corresponden a neumonía en {reg}.",
        "Conclusión radiológica: Lo más probable es un proceso neumónico activo en {reg}.",
        "Impresión diagnóstica: Probable neumonía adquirida en la comunidad en {reg}, correlacionar con clínica.",
        "Conclusión: Infiltrado focal en {reg}, probablemente neumónico bacteriano o atípico.",
    ]

    # 3. Posible: infiltrado "posible", "dudoso" o "no se puede descartar"
    pos_hall = [
        "Tenue opacidad mal definida en {reg}, parcialmente superpuesta a arcos costales.",
        "Discreto infiltrado intersticial inespecífico en {reg}, sin consolidación franca.",
        "Sutil aumento de densidad basal en {reg} con técnica en baja inspiración.",
        "Alteración parenquimatosa dudosa en {reg}, de aspecto inespecífico.",
    ]
    pos_conc = [
        "Impresión diagnóstica: Posible infiltrado neumónico incipiente en {reg}; no se puede descartar neumonía inicial.",
        "Conclusión radiológica: Hallazgo dudoso en {reg}, no se descarta proceso infeccioso temprano.",
        "Impresión diagnóstica: Infiltrado posible en {reg} versus atelectasia laminar; sugerimos control si persisten síntomas.",
        "Conclusión: Opacidad tenue dudosa; posible neumonía atípica inicial, correlacionar con clínica.",
    ]

    # 4. Sin neumonía: normal o con otra alteración (atelectasia, cicatriz)
    neg_hall = [
        "Campos pulmonares bien ventilados y simétricos, sin evidencia de infiltrados alveolares ni consolidaciones.",
        "Parénquima pulmonar limpio y despejado. No se observan opacidades ni focos condensantes.",
        "Pequeña banda fibrocicatricial antigua en {reg}, sin infiltrados agudos ni signos de infección.",
        "Mínima atelectasia laminar basal en {reg}, sin consolidaciones parenquimatosas ni derrame pleural.",
        "Leve engrosamiento bronquial difuso crónico compatible con EPOC, ausencia de focos neumónicos agudos.",
    ]
    neg_conc = [
        "Impresión diagnóstica: Radiografía de tórax normal, sin evidencia de neumonía.",
        "Conclusión radiológica: Estudio sin signos de neumonía. Se descarta compromiso infeccioso agudo.",
        "Impresión diagnóstica: Cambios cicatriciales crónicos en {reg}. Sin neumonía en el examen actual.",
        "Conclusión: Atelectasia subsegmentaria leve sin evidencia de neumonía.",
    ]

    # Casos límite realistas (~8% de solapamiento entre clases vecinas)
    border_def_prob = "Consolidación parcheada moderada en {reg}; hallazgos altamente sugestivos de foco neumónico en evolución."
    border_prob_pos = "Opacidad focal tenue en {reg} que podría corresponder a proceso inflamatorio o probable foco incipiente."
    border_pos_neg = "Baja inspiración con discreto refuerzo de trama basal en {reg}; sin consolidación franca, aunque difícil descartar foco muy inicial."

    per_cls = n_samples // 4
    classes = ["Sin neumonía"] * per_cls + ["Posible"] * per_cls + ["Probable"] * per_cls + ["Definitiva"] * (n_samples - 3 * per_cls)
    rng.shuffle(classes)

    records = []
    for idx, label in enumerate(classes, start=1):
        reg = rng.choice(regiones)
        h = rng.choice(headers)
        m = rng.choice(motivos)
        ex1 = rng.choice(normales_extra)
        ex2 = rng.choice([e for e in normales_extra if e != ex1])
        r = rng.random()

        if label == "Definitiva":
            if r < 0.08:
                texto = f"{h} {m} Hallazgos: {border_def_prob.format(reg=reg)} {ex1}"
            else:
                texto = f"{h} {m} Hallazgos: {rng.choice(def_hall).format(reg=reg)} {ex1} {ex2} {rng.choice(def_conc).format(reg=reg)}"
        elif label == "Probable":
            if r < 0.06:
                texto = f"{h} {m} Hallazgos: {border_def_prob.format(reg=reg)} {ex1}"
            elif r < 0.12:
                texto = f"{h} {m} Hallazgos: {border_prob_pos.format(reg=reg)} {ex1}"
            else:
                texto = f"{h} {m} Hallazgos: {rng.choice(prob_hall).format(reg=reg)} {ex1} {ex2} {rng.choice(prob_conc).format(reg=reg)}"
        elif label == "Posible":
            if r < 0.06:
                texto = f"{h} {m} Hallazgos: {border_prob_pos.format(reg=reg)} {ex1}"
            elif r < 0.12:
                texto = f"{h} {m} Hallazgos: {border_pos_neg.format(reg=reg)} {ex1}"
            else:
                texto = f"{h} {m} Hallazgos: {rng.choice(pos_hall).format(reg=reg)} {ex1} {ex2} {rng.choice(pos_conc).format(reg=reg)}"
        else:  # Sin neumonía
            if r < 0.08:
                texto = f"{h} {m} Hallazgos: {border_pos_neg.format(reg=reg)} {ex1}"
            else:
                texto = f"{h} {m} Hallazgos: {rng.choice(neg_hall).format(reg=reg)} {ex1} {ex2} {rng.choice(neg_conc).format(reg=reg)}"

        records.append(
            {
                "report_id": f"RX-{idx:04d}",
                "tipo_documento": "INFORME RADIOLOGICO",
                "texto_informe": texto,
                "categoria_rx": label,
            }
        )

    return pd.DataFrame(records)


def build_sample_pdf(
    output_path: Path,
    titulo: str,
    paciente_rut: str,
    paciente_edad: str,
    fecha: str,
    tipo_doc: str,
    motivo: str,
    hallazgos: list[str],
    conclusion: str,
    medico: str,
) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=2.0 * cm,
        leftMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleMed", parent=styles["Heading1"], fontSize=14.5, textColor=colors.HexColor("#0f3d5e"), spaceAfter=6)
    sub_style = ParagraphStyle("SubMed", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#475569"), spaceAfter=12)
    sec_style = ParagraphStyle("SecMed", parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#0f3d5e"), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("BodyMed", parent=styles["Normal"], fontSize=10.5, leading=15, textColor=colors.HexColor("#1e293b"), spaceAfter=6)

    story = [
        Paragraph("HOSPITAL DEMOSTRATIVO NEUMOALERT · SERVICIO DE IMAGENOLOGÍA", sub_style),
        Paragraph(titulo, title_style),
        Paragraph("Documento clínico ficticio alineado a las 4 categorías del estudio Heckerling (Sección E).", sub_style),
    ]
    info_table = Table(
        [
            ["Tipo de documento:", tipo_doc, "Fecha:", fecha],
            ["RUT Paciente (ficticio):", paciente_rut, "Edad:", paciente_edad],
        ],
        colWidths=[4.2 * cm, 5.2 * cm, 2.2 * cm, 5.2 * cm],
    )
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([
        info_table,
        Spacer(1, 10),
        Paragraph("MOTIVO DE ESTUDIO", sec_style),
        Paragraph(motivo, body_style),
        Paragraph("HALLAZGOS RADIOLÓGICOS", sec_style),
        *[Paragraph(f"• {h}", body_style) for h in hallazgos],
        Paragraph("IMPRESIÓN DIAGNÓSTICA / CONCLUSIÓN RADIOLÓGICA", sec_style),
        Paragraph(f"<b>{conclusion}</b>", body_style),
        Spacer(1, 22),
        Paragraph(f"Profesional informante (ficticio): {medico}", sub_style),
    ])
    doc.build(story)


def generate_sample_pdfs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    # Limpiar PDFs anteriores para mantener exactamente los 5 casos alineados al estudio
    for old_pdf in PDF_DIR.glob("*.pdf"):
        old_pdf.unlink()

    build_sample_pdf(
        PDF_DIR / "01_rx_definitiva_neumonia.pdf",
        titulo="INFORME RADIOLÓGICO: RADIOGRAFÍA DE TÓRAX PA Y LATERAL",
        paciente_rut="16482910-3",
        paciente_edad="68 años",
        fecha="24-11-2026",
        tipo_doc="INFORME RADIOLÓGICO",
        motivo="Paciente adulto con fiebre 39,0 °C, FC 112 lpm, crepitaciones y murmullo disminuido derecho (Heckerling 5/5). Descartar neumonía.",
        hallazgos=[
            "Extensa consolidación parenquimatosa homogénea en lóbulo inferior derecho con broncograma aéreo evidente en su interior.",
            "Opacidad alveolar confluente basal derecha asociada a mínimo derrame pleural ipsilateral.",
            "Campo pulmonar izquierdo despejado. Silueta cardiomediastínica conservada.",
        ],
        conclusion="Neumonía confirmada radiográficamente en lóbulo inferior derecho. Infiltrado neumónico inequívoco (Categoría: Definitiva).",
        medico="Dra. Sofía Valenzuela (Radiología - Demo)",
    )

    build_sample_pdf(
        PDF_DIR / "02_rx_probable_neumonia.pdf",
        titulo="INFORME RADIOLÓGICO: RADIOGRAFÍA DE TÓRAX AP Y LATERAL",
        paciente_rut="17324509-2",
        paciente_edad="59 años",
        fecha="24-11-2026",
        tipo_doc="INFORME RADIOLÓGICO",
        motivo="Cuadro de 5 días de tos, fiebre 38,3 °C (tomó paracetamol hace 3 horas), SatO2 92% y crepitaciones basales izquierdas.",
        hallazgos=[
            "Opacidad alveolar parcheada en base pulmonar izquierda de densidad moderada, sin derrame pleural cuantioso.",
            "Discreto refuerzo peribronquial adyacente con leve borramiento vascular local.",
            "Campo pulmonar derecho sin consolidaciones evidentes.",
        ],
        conclusion="Hallazgos que probablemente corresponden a neumonía adquirida en la comunidad en base izquierda; lo más probable es un proceso neumónico activo (Categoría: Probable).",
        medico="Dr. Martín Herrera (Radiología - Demo)",
    )

    build_sample_pdf(
        PDF_DIR / "03_rx_posible_dudoso.pdf",
        titulo="INFORME RADIOLÓGICO: RADIOGRAFÍA DE TÓRAX FRONTAL",
        paciente_rut="20874123-3",
        paciente_edad="42 años",
        fecha="24-11-2026",
        tipo_doc="INFORME RADIOLÓGICO",
        motivo="Sospecha clínica de neumonía en urgencia, febrícula y tos de 4 días.",
        hallazgos=[
            "Tenue infiltrado intersticial inespecífico en base derecha de bordes mal definidos, parcialmente oscurecido por baja inspiración.",
            "No se observa consolidación lobar franca ni derrame pleural.",
            "Silueta cardíaca normal. Senos costofrénicos libres.",
        ],
        conclusion="Posible infiltrado neumónico incipiente en base derecha; hallazgo dudoso, no se puede descartar neumonía inicial. Correlacionar con clínica (Categoría: Posible).",
        medico="Dra. Camila Rojas (Radiología - Demo)",
    )

    build_sample_pdf(
        PDF_DIR / "04_rx_sin_neumonia_con_negaciones.pdf",
        titulo="INFORME RADIOLÓGICO: RADIOGRAFÍA DE TÓRAX PA Y LATERAL",
        paciente_rut="19874562-1",
        paciente_edad="31 años",
        fecha="24-11-2026",
        tipo_doc="INFORME RADIOLÓGICO",
        motivo="Derivado desde urgencia por sospecha de neumonía, tos y rinorrea/odinofagia de 3 días. Descartar infiltrados.",
        hallazgos=[
            "Campos pulmonares bien ventilados y simétricos, sin evidencia de infiltrados alveolares ni consolidaciones.",
            "Mínima atelectasia laminar basal sin compromiso parenquimatoso agudo.",
            "Ausencia de derrame pleural y neumotórax. Silueta cardiomediastínica normal.",
        ],
        conclusion="Radiografía de tórax sin evidencia de neumonía. Se descarta compromiso infeccioso agudo en el estudio actual (Categoría: Sin neumonía).",
        medico="Dra. Camila Rojas (Radiología - Demo)",
    )

    build_sample_pdf(
        PDF_DIR / "05_antecedente_epicrisis_previa.pdf",
        titulo="EPICRISIS / RESUMEN DE HOSPITALIZACIÓN PREVIA",
        paciente_rut="14390218-8",
        paciente_edad="74 años",
        fecha="14-05-2026",
        tipo_doc="EPICRISIS",
        motivo="Resumen de hospitalización previa por descompensación de EPOC y antecedente respiratorio.",
        hallazgos=[
            "Antecedente de neumonía basal izquierda tratada hace 6 meses con resolución clínica satisfactoria.",
            "Radiografía al alta mostró cambios cicatriciales antiguos leves sin consolidación activa.",
        ],
        conclusion="Alta médica con resolución del cuadro previo. Portador de EPOC en control ambulatorio.",
        medico="Dr. Rodrigo Fuentes (Medicina Interna - Demo)",
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    df_clin = generate_heckerling_clinical_dataset(n_samples=2500, seed=42)
    clin_path = DATA_DIR / "clinical_pneumonia_dataset.csv"
    df_clin.to_csv(clin_path, index=False, encoding="utf-8-sig")

    print(f"Dataset clínico Heckerling guardado: {clin_path} ({len(df_clin)} filas)")
    print(f"Prevalencia RxT alterada (Probable+Definitiva): {df_clin['rx_alterada'].mean():.1%}")
    print("\nDistribución Categoría Rx (Sección E):")
    print(df_clin["categoria_rx"].value_counts().to_string())

    print("\nTabla por Puntaje de Heckerling (0 a 5) en nuestro dataset:")
    tabla_h = df_clin.groupby("puntaje_heckerling")["rx_alterada"].agg(["count", "mean"])
    tabla_h["pct_alterada"] = (tabla_h["mean"] * 100).round(1)
    print(tabla_h[["count", "pct_alterada"]].to_string())

    df_rx = generate_radiology_reports_4classes(n_samples=1000, seed=42)
    rx_path = DATA_DIR / "radiology_reports_dataset.csv"
    df_rx.to_csv(rx_path, index=False, encoding="utf-8-sig")
    print(f"\nDataset informes Rx (4 categorías) guardado: {rx_path} ({len(df_rx)} filas)")
    print(df_rx["categoria_rx"].value_counts().to_string())

    generate_sample_pdfs()
    print(f"\n5 PDFs de prueba generados en: {PDF_DIR}")


if __name__ == "__main__":
    main()
