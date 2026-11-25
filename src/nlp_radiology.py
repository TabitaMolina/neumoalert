"""
Módulo de NLP Clínico para informes radiológicos PDF de NEUMOALERT.

Alineado exactamente a las 4 categorías de la Sección E del estudio del médico:
  1. Sin neumonía — Normal, o con otra alteración (atelectasia, cicatriz)
  2. Posible      — Infiltrado "posible", "dudoso" o "no se puede descartar"
  3. Probable     — "Probablemente" o "lo más probable" es neumonía
  4. Definitiva   — Infiltrado neumónico inequívoco

Incluye:
- Extracción de texto con PyMuPDF (y respaldo OCR opcional si pytesseract está disponible).
- Filtro automático de guías/protocolos teóricos (no corresponden a un paciente).
- Detección de palabras clave radiológicas con manejo de negaciones y contexto.
- Modelo NLP entrenado (TF-IDF + Regresión Logística) en las 4 categorías exactas.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import joblib
import pymupdf as fitz  # PyMuPDF
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
NLP_MODEL_PATH = BASE_DIR / "models" / "nlp_radiology_model.joblib"

_NLP_CACHE: dict[str, Any] | None = None

CATEGORIAS_ORDEN = ["Sin neumonía", "Posible", "Probable", "Definitiva"]

RADIOLOGY_KEYWORDS = [
    ("neumonía / foco neumónico", r"\b(?:neumon[ií]as?|neum[oó]nic[oa]s?|bronconeumon[ií]a)\b"),
    ("consolidación / condensación", r"\b(?:consolidaci[oó]n(?:es)?|condensaci[oó]n(?:es)?|condensantes?)\b"),
    ("infiltrado pulmonar", r"\b(?:infiltrados?(?:\s+alveolar(?:es)?|\s+intersticial(?:es)?|\s+pulmonar(?:es)?|\s+parenquimatosos?|\s+neum[oó]nicos?)?)\b"),
    ("opacidad alveolar", r"\b(?:opacidad(?:es)?(?:\s+alveolar(?:es)?|\s+algodonosa|\s+homog[eé]nea|\s+tenue|\s+dudosa|\s+focal)?)\b"),
    ("broncograma aéreo", r"\bbroncograma\s+a[eé]reo\b"),
    ("derrame pleural", r"\bderrame\s+pleural\b"),
    ("relleno alveolar", r"\brelleno\s+alveolar\b"),
    ("atelectasia / cicatriz", r"\b(?:atelectasia|cicatricial(?:es)?|fibrocicatricial(?:es)?)\b"),
]

NEGATION_PATTERNS = [
    r"\bsin\s+evidencia\s+de\b",
    r"\bsin\s+signos\s+(?:radiol[oó]gicos\s+)?de\b",
    r"\bno\s+se\s+(?:observan?|evidencian?|identifican?|aprecian?|visualizan?)\b",
    r"\bno\s+hay\s+(?:evidencia\s+de|signos\s+de|infiltrados|consolidaci[oó]n|neumon[ií]a)\b",
    r"\bausencia\s+de\b",
    r"\bse\s+descarta\b",
    r"\bsin\s+neumon[ií]a\b",
    r"\bsin\s+alteraciones\s+(?:pleuroparenquimatosas\s+)?agudas\b",
    r"\bcampos\s+pulmonares\s+(?:bien\s+ventilados|limpios|despejados)\b",
    r"\bradiograf[ií]a\s+de\s+t[oó]rax\s+normal\b",
]

POSIBLE_PATTERNS = [
    r"\bno\s+se\s+(?:puede\s+)?descartar?\b",
    r"\bno\s+puede\s+descartarse\b",
    r"\bposibles?\b",
    r"\bdudos[oa]s?\b",
    r"\bincipientes?\b",
    r"\binespec[ií]fic[oa]s?\b",
    r"\btenues?\b",
    r"\bpodr[ií]a\s+corresponder\b",
]

PROBABLE_PATTERNS = [
    r"\bprobablemente\b",
    r"\blo\s+m[aá]s\s+probable\b",
    r"\bprobable\s+(?:neumon[ií]a|foco|proceso)\b",
    r"\baltamente\s+sugestiv[oa]s?\b",
    r"\bopacidad\s+alveolar\s+parcheada\b",
]

DEFINITIVA_PATTERNS = [
    r"\binequ[ií]voc[oa]\b",
    r"\bneumon[ií]a\s+confirmada\b",
    r"\bconsolidaci[oó]n\s+neum[oó]nica\s+definitiva\b",
    r"\bcuadro\s+radiogr[aá]fico\s+categ[oó]rico\b",
    r"\bconsolidaci[oó]n\s+parenquimatosa\s+homog[eé]nea\b",
    r"\bbroncograma\s+a[eé]reo\b",
    r"\binfiltrado\s+neum[oó]nico\s+lobar\s+franco\b",
    r"\bevidencia\s+clara\s+e\s+inequ[ií]voca\b",
]

MOTIVE_CONTEXT_PATTERNS = [
    r"^(?:motivo(?:\s+de\s+(?:estudio|solicitud|consulta))?|antecedente\s+cl[ií]nico|indicaci[oó]n|indicaciones?|derivad[oa]\s+por|paciente\s+adulto)\b",
    r"\bdescartar\s+(?:neumon[ií]a|infiltrados|foco\s+neum[oó]nico)\b",
    r"\bevaluar\s+(?:presencia\s+de|infiltrados|focos)\b",
]


def reload_nlp_cache() -> None:
    global _NLP_CACHE
    _NLP_CACHE = None


def load_nlp_bundle() -> dict[str, Any] | None:
    global _NLP_CACHE
    if _NLP_CACHE is None and NLP_MODEL_PATH.exists():
        _NLP_CACHE = joblib.load(NLP_MODEL_PATH)
    return _NLP_CACHE


def extraer_texto_pdf(pdf_source: bytes | str | Path) -> dict[str, Any]:
    if isinstance(pdf_source, (str, Path)):
        doc = fitz.open(str(pdf_source))
    else:
        doc = fitz.open(stream=pdf_source, filetype="pdf")

    texto_paginas = []
    uso_ocr = False
    ocr_no_disponible = False

    for pagina in doc:
        texto = pagina.get_text("text") or ""
        if len(texto.strip()) < 40:
            try:
                import pytesseract  # type: ignore

                pix = pagina.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                texto_ocr = pytesseract.image_to_string(img, lang="spa")
                if texto_ocr.strip():
                    texto = texto_ocr
                    uso_ocr = True
            except Exception:
                ocr_no_disponible = True
        texto_paginas.append(texto)

    num_paginas = len(doc)
    doc.close()

    return {
        "texto": "\n".join(texto_paginas).strip(),
        "num_paginas": num_paginas,
        "uso_ocr": uso_ocr,
        "ocr_no_disponible": ocr_no_disponible,
    }


def identificar_documento(texto: str) -> str:
    if not texto or not texto.strip():
        return "SIN DOCUMENTO PDF"
    t = texto.lower()

    marcadores_guia = [
        "protocolo de práctica clínica",
        "guía de diagnóstico y tratamiento",
        "guía clínica",
        "protocolo de manejo",
        "bibliografía",
        "abstract",
        "epidemiología",
        "palabras claves:",
        "criterios de hospitalización",
    ]
    if sum(1 for m in marcadores_guia if m in t) >= 2 or "protocolo de práctica clínica" in t or "guía de diagnóstico y tratamiento" in t:
        return "PROTOCOLO O GUÍA CLÍNICA"

    if "epicrisis" in t or "resumen de hospitalización" in t:
        return "EPICRISIS"

    if any(k in t for k in ["informe radiológico", "informe radiologico", "impresión diagnóstica", "hallazgos radiológicos", "conclusión radiológica", "radiografía de tórax"]):
        return "INFORME RADIOLOGICO"

    if any(k in t for k in ["informe médico", "evolución clínica", "anamnesis", "urgencia"]):
        return "INFORME MEDICO INDIVIDUAL"

    return "DOCUMENTO NO IDENTIFICADO"


def _es_oracion_motivo(oracion: str) -> bool:
    return any(re.search(p, oracion) for p in MOTIVE_CONTEXT_PATTERNS)


def extraer_hallazgos_reglas(texto: str, tipo_documento: str) -> dict[str, Any]:
    if tipo_documento in ("PROTOCOLO O GUÍA CLÍNICA", "SIN DOCUMENTO PDF"):
        return {
            "hallazgos": [],
            "palabras_clave_detectadas": [],
            "categoria_reglas": "NO CLASIFICABLE" if tipo_documento == "PROTOCOLO O GUÍA CLÍNICA" else "SIN INFORME PDF",
            "conteo": {"Definitiva": 0, "Probable": 0, "Posible": 0, "Sin neumonía": 0, "Contexto": 0},
        }

    texto_norm = re.sub(r"[•·]+", ". ", texto)
    texto_norm = re.sub(r"\r\n?", "\n", texto_norm)
    texto_norm = re.sub(r"\n{2,}", ". ", texto_norm)
    texto_norm = re.sub(r"\n", " ", texto_norm)
    texto_norm = re.sub(r"\s+", " ", texto_norm).strip()

    oraciones = [o.strip() for o in re.split(r"(?<=[.;])\s+|\s*;\s*", texto_norm) if len(o.strip()) >= 8]

    hallazgos = []
    kw_set = set()
    conteo = {"Definitiva": 0, "Probable": 0, "Posible": 0, "Sin neumonía": 0, "Contexto": 0}

    for oracion in oraciones:
        ol = oracion.lower()
        kws = [nombre for nombre, patron in RADIOLOGY_KEYWORDS if re.search(patron, ol)]
        tiene_neg = any(re.search(p, ol) for p in NEGATION_PATTERNS)

        if not kws and not tiene_neg:
            continue

        if _es_oracion_motivo(ol) and not any(x in ol for x in ["impresión diagnóstica", "conclusión", "se observa", "campos pulmonares"]):
            estado = "Contexto"
        elif any(re.search(p, ol) for p in POSIBLE_PATTERNS):
            estado = "Posible"
        elif tiene_neg:
            estado = "Sin neumonía"
        elif any(re.search(p, ol) for p in DEFINITIVA_PATTERNS):
            estado = "Definitiva"
        elif any(re.search(p, ol) for p in PROBABLE_PATTERNS):
            estado = "Probable"
        else:
            estado = "Contexto"

        for k in kws:
            kw_set.add((k, estado))

        conteo[estado] += 1
        hallazgos.append({"oracion": oracion, "estado": estado, "palabras_clave": kws})

    if conteo["Definitiva"] > 0:
        cat_reglas = "Definitiva"
    elif conteo["Probable"] > 0:
        cat_reglas = "Probable"
    elif conteo["Posible"] > 0:
        cat_reglas = "Posible"
    elif conteo["Sin neumonía"] > 0:
        cat_reglas = "Sin neumonía"
    else:
        cat_reglas = "Sin neumonía"

    orden_pol = {"Definitiva": 0, "Probable": 1, "Posible": 2, "Sin neumonía": 3, "Contexto": 4}
    palabras_clave_resumen = [
        {"termino": term, "polaridad": pol}
        for term, pol in sorted(kw_set, key=lambda x: (orden_pol.get(x[1], 5), x[0]))
    ]

    return {
        "hallazgos": hallazgos,
        "palabras_clave_detectadas": palabras_clave_resumen,
        "categoria_reglas": cat_reglas,
        "conteo": conteo,
    }


def analizar_informe_nlp(texto: str) -> dict[str, Any]:
    """Clasifica el informe en las 4 categorías exactas de la Sección E + probabilidad continua."""
    tipo_documento = identificar_documento(texto)
    reglas = extraer_hallazgos_reglas(texto, tipo_documento)

    if tipo_documento in ("PROTOCOLO O GUÍA CLÍNICA", "SIN DOCUMENTO PDF"):
        return {
            "es_valido_paciente": False,
            "es_antecedente_previo": False,
            "tipo_documento": tipo_documento,
            "prob_nlp": None,
            "clase_nlp": reglas["categoria_reglas"],
            "probabilidades_clases": {},
            "hallazgos": [],
            "palabras_clave": [],
            "resumen": (
                "Este archivo es una guía médica general, no el informe de un paciente. Se bloqueó automáticamente."
                if tipo_documento == "PROTOCOLO O GUÍA CLÍNICA"
                else "No se cargó informe PDF."
            ),
        }

    bundle = load_nlp_bundle()
    probs_clases: dict[str, float] = {}
    prob_ml: float | None = None
    clase_ml: str | None = None

    if bundle is not None:
        pipeline = bundle["pipeline"]
        classes = list(pipeline.classes_)
        raw_probs = pipeline.predict_proba([texto])[0]
        probs_clases = {c: round(float(p), 4) for c, p in zip(classes, raw_probs)}
        clase_ml = max(probs_clases, key=probs_clases.get)
        # Probabilidad continua ponderando las 4 categorías del estudio Heckerling
        prob_ml = (
            0.96 * probs_clases.get("Definitiva", 0.0)
            + 0.78 * probs_clases.get("Probable", 0.0)
            + 0.42 * probs_clases.get("Posible", 0.0)
            + 0.04 * probs_clases.get("Sin neumonía", 0.0)
        )

    mapa_reglas_prob = {
        "Definitiva": 0.95,
        "Probable": 0.78,
        "Posible": 0.45,
        "Sin neumonía": 0.06,
    }
    prob_reglas = mapa_reglas_prob.get(reglas["categoria_reglas"], 0.15)

    if tipo_documento == "EPICRISIS":
        prob_final = round(0.50 * (prob_ml if prob_ml is not None else prob_reglas) + 0.50 * prob_reglas, 4)
        clase_final = "Antecedente previo (Epicrisis)"
        resumen = "Epicrisis antigua detectada: se usa solo como antecedente histórico, no como diagnóstico actual."
    else:
        prob_final = round(min(max(0.65 * prob_ml + 0.35 * prob_reglas, 0.02), 0.99), 4) if prob_ml is not None else round(prob_reglas, 4)
        if prob_final >= 0.85:
            clase_final = "Definitiva"
            resumen = "El informe describe un infiltrado neumónico inequívoco (categoría Definitiva)."
        elif prob_final >= 0.62:
            clase_final = "Probable"
            resumen = "El informe indica que lo más probable es neumonía (categoría Probable)."
        elif prob_final >= 0.30:
            clase_final = "Posible"
            resumen = "El informe menciona hallazgos dudosos o que no descartan neumonía (categoría Posible)."
        else:
            clase_final = "Sin neumonía"
            resumen = "El informe descarta neumonía o muestra hallazgos normales/crónicos (categoría Sin neumonía)."

    return {
        "es_valido_paciente": True,
        "es_antecedente_previo": tipo_documento == "EPICRISIS",
        "tipo_documento": tipo_documento,
        "prob_nlp": prob_final,
        "clase_nlp": clase_final,
        "clase_ml": clase_ml,
        "probabilidades_clases": probs_clases,
        "categoria_reglas": reglas["categoria_reglas"],
        "hallazgos": reglas["hallazgos"],
        "palabras_clave": reglas["palabras_clave_detectadas"],
        "resumen": resumen,
    }


def analizar_pdf_completo(pdf_source: bytes | str | Path) -> dict[str, Any]:
    ext = extraer_texto_pdf(pdf_source)
    analisis = analizar_informe_nlp(ext["texto"])
    analisis["texto_extraido"] = ext["texto"]
    analisis["num_paginas"] = ext["num_paginas"]
    analisis["uso_ocr"] = ext["uso_ocr"]
    return analisis
