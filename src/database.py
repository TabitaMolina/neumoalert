"""
Base de datos SQLAlchemy (SQLite local por defecto / PostgreSQL vía DATABASE_URL)
para NEUMOALERT v3, alineada a la ficha médica de Heckerling y las 4 categorías del radiólogo.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "neumoalert_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'neumoalert_v3.db'}")

metadata = MetaData()

pacientes = Table(
    "pacientes",
    metadata,
    Column("id", String(15), primary_key=True),  # RUT chileno ficticio
    Column("edad", Integer, nullable=False),
    Column("sexo", String(10), nullable=False),
    Column("dias_evolucion", Integer, default=4),
    # Sección B: Enfermedades previas
    Column("asma", Boolean, default=False),
    Column("epoc", Boolean, default=False),
    Column("otra_enf_pulmonar", Boolean, default=False),
    Column("insuficiencia_cardiaca", Boolean, default=False),
    Column("inmunosupresion", Boolean, default=False),
    Column("demencia", Boolean, default=False),
    # Sección C: Heckerling
    Column("temperatura", Float),
    Column("frecuencia_cardiaca", Integer),
    Column("crepitantes", Boolean, default=False),
    Column("murmullo_disminuido", Boolean, default=False),
    Column("puntaje_heckerling", Integer, default=0),
    # Sección D: Datos adicionales del médico
    Column("frecuencia_respiratoria", Integer),
    Column("sat_o2", Float),
    Column("matidez", Boolean, default=False),
    Column("egofonia", Boolean, default=False),
    Column("soplo_tubarico", Boolean, default=False),
    Column("compromiso_conciencia", Boolean, default=False),
    Column("rinorrea_odinofagia", Boolean, default=False),
    Column("antipiretico_6h", Boolean, default=False),
    # Resultados IA
    Column("prob_rf", Float),
    Column("prob_cnn", Float),
    Column("prob_nlp", Float),
    Column("prob_integrada", Float),
    Column("categoria_rx_nlp", String(40), default="Sin informe"),
    Column("concordancia_ia", String(60), default="No evaluada"),
    Column("prioridad", String(20), default="VERDE"),
    Column("fecha_ingreso", DateTime, nullable=False),
    Column("estado", String(40), default="En evaluación"),
)

imagenes = Table(
    "imagenes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("paciente_id", String(15), ForeignKey("pacientes.id"), nullable=False),
    Column("ruta_imagen", Text),
    Column("tipo", String(40)),
    Column("prob_ia", Float),
    Column("prediccion_ia", String(120)),
    Column("modo_cnn", String(50)),
    Column("fecha", DateTime, nullable=False),
)

diagnosticos = Table(
    "diagnosticos",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("paciente_id", String(15), ForeignKey("pacientes.id"), nullable=False),
    Column("tipo_documento", String(60)),
    Column("texto_medico", Text),
    Column("categoria_rx", String(40)),  # Sin neumonía / Posible / Probable / Definitiva
    Column("prob_nlp", Float),
    Column("palabras_clave", Text),
    Column("diagnostico_final", Text),
    Column("concordancia", String(60)),
    Column("pdf_path", Text),
    Column("fecha", DateTime, nullable=False),
)

alertas = Table(
    "alertas",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("paciente_id", String(15), ForeignKey("pacientes.id"), nullable=False),
    Column("prioridad", String(20), nullable=False),
    Column("motivo", Text),
    Column("estado_alerta", String(30), default="Pendiente"),
    Column("fecha", DateTime, nullable=False),
)


def create_db_engine():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def rut_dv(numero: int | str) -> str:
    total = 0
    factor = 2
    for digito in reversed(str(numero)):
        total += int(digito) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - (total % 11)
    if resto == 11:
        return "0"
    if resto == 10:
        return "K"
    return str(resto)


def rut_aleatorio(rng: random.Random | None = None) -> str:
    r = rng or random
    numero = r.randint(14_000_000, 27_500_000)
    return f"{numero}-{rut_dv(numero)}"


def guardar_archivo(uploaded_file, rut: str, categoria: str) -> str | None:
    if uploaded_file is None:
        return None
    extension = Path(uploaded_file.name).suffix.lower()
    nombre = f"{rut}_{categoria}_{uuid.uuid4().hex[:8]}{extension}"
    destino = UPLOAD_DIR / nombre
    destino.write_bytes(uploaded_file.getbuffer())
    return str(destino)


def obtener_registros(engine, tabla: Table) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(select(tabla), conn)


def insertar_casos_demo_si_vacia(engine, rf_predictor, fusion_fn) -> None:
    """Crea 8 casos clínicos realistas (adultos >= 18 años) usando exactamente la ficha Heckerling."""
    with engine.begin() as conn:
        if conn.execute(select(pacientes.c.id)).first() is not None:
            return

        casos_demo = [
            {
                "rut": "16482910-3", "edad": 71, "sexo": "M", "dias_evolucion": 4,
                "asma": False, "epoc": True, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
                "temperatura": 39.1, "frecuencia_cardiaca": 114, "crepitantes": True, "murmullo_disminuido": True,
                "frecuencia_respiratoria": 30, "sat_o2": 88.0, "matidez": True, "egofonia": False, "soplo_tubarico": True,
                "compromiso_conciencia": True, "rinorrea_odinofagia": False, "antipiretico_6h": False,
                "prob_cnn": 0.94, "prob_nlp": 0.95, "cat_rx": "Definitiva", "tipo_doc": "INFORME RADIOLOGICO",
                "texto_med": "Consolidación parenquimatosa homogénea en lóbulo inferior derecho con broncograma aéreo. Infiltrado neumónico inequívoco.",
                "keywords": "consolidación (Definitiva), broncograma aéreo (Definitiva), neumonía (Definitiva)",
                "estado": "En evaluación", "min_atras": 14,
            },
            {
                "rut": "19204511-8", "edad": 64, "sexo": "F", "dias_evolucion": 5,
                "asma": False, "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
                "temperatura": 38.4, "frecuencia_cardiaca": 108, "crepitantes": True, "murmullo_disminuido": True,
                "frecuencia_respiratoria": 26, "sat_o2": 91.0, "matidez": False, "egofonia": False, "soplo_tubarico": False,
                "compromiso_conciencia": False, "rinorrea_odinofagia": False, "antipiretico_6h": False,
                "prob_cnn": 0.86, "prob_nlp": 0.08, "cat_rx": "Sin neumonía", "tipo_doc": "INFORME RADIOLOGICO",
                "texto_med": "Lectura preliminar de urgencia: sin evidencia de infiltrados ni neumonía en bases pulmonares.",
                "keywords": "infiltrado pulmonar (Sin neumonía), neumonía (Sin neumonía)",
                "estado": "Pendiente de revisión", "min_atras": 28,
            },
            {
                "rut": "17324509-2", "edad": 58, "sexo": "F", "dias_evolucion": 5,
                "asma": False, "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": True, "demencia": False,
                "temperatura": 37.4, "frecuencia_cardiaca": 106, "crepitantes": True, "murmullo_disminuido": False,
                "frecuencia_respiratoria": 25, "sat_o2": 92.0, "matidez": False, "egofonia": False, "soplo_tubarico": False,
                "compromiso_conciencia": False, "rinorrea_odinofagia": False, "antipiretico_6h": True,
                "prob_cnn": 0.81, "prob_nlp": 0.78, "cat_rx": "Probable", "tipo_doc": "INFORME RADIOLOGICO",
                "texto_med": "Opacidad alveolar parcheada en base izquierda. Hallazgos que probablemente corresponden a neumonía.",
                "keywords": "opacidad alveolar (Probable), neumonía (Probable)",
                "estado": "En evaluación", "min_atras": 45,
            },
            {
                "rut": "20874123-3", "edad": 43, "sexo": "F", "dias_evolucion": 4,
                "asma": False, "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
                "temperatura": 37.9, "frecuencia_cardiaca": 96, "crepitantes": False, "murmullo_disminuido": False,
                "frecuencia_respiratoria": 21, "sat_o2": 95.0, "matidez": False, "egofonia": False, "soplo_tubarico": False,
                "compromiso_conciencia": False, "rinorrea_odinofagia": False, "antipiretico_6h": False,
                "prob_cnn": 0.54, "prob_nlp": 0.45, "cat_rx": "Posible", "tipo_doc": "INFORME RADIOLOGICO",
                "texto_med": "Tenue infiltrado intersticial basal derecho; hallazgo dudoso, no se puede descartar neumonía inicial.",
                "keywords": "infiltrado pulmonar (Posible), neumonía (Posible)",
                "estado": "Pendiente de informe", "min_atras": 75,
            },
            {
                "rut": "14390218-8", "edad": 76, "sexo": "M", "dias_evolucion": 3,
                "asma": False, "epoc": True, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": True, "inmunosupresion": False, "demencia": False,
                "temperatura": 37.1, "frecuencia_cardiaca": 98, "crepitantes": True, "murmullo_disminuido": False,
                "frecuencia_respiratoria": 24, "sat_o2": 93.0, "matidez": False, "egofonia": False, "soplo_tubarico": False,
                "compromiso_conciencia": False, "rinorrea_odinofagia": False, "antipiretico_6h": False,
                "prob_cnn": 0.48, "prob_nlp": None, "cat_rx": "Sin informe", "tipo_doc": "SIN DOCUMENTO PDF",
                "texto_med": "Ingreso en Modo Triage (a la espera de informe radiológico). Portador de EPOC e ICC.",
                "keywords": "",
                "estado": "En evaluación", "min_atras": 96,
            },
            {
                "rut": "18765432-7", "edad": 32, "sexo": "M", "dias_evolucion": 3,
                "asma": True, "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
                "temperatura": 36.8, "frecuencia_cardiaca": 106, "crepitantes": False, "murmullo_disminuido": False,
                "frecuencia_respiratoria": 23, "sat_o2": 96.0, "matidez": False, "egofonia": False, "soplo_tubarico": False,
                "compromiso_conciencia": False, "rinorrea_odinofagia": True, "antipiretico_6h": False,
                "prob_cnn": 0.16, "prob_nlp": 0.06, "cat_rx": "Sin neumonía", "tipo_doc": "INFORME RADIOLOGICO",
                "texto_med": "Crisis asmática leve. Campos pulmonares bien ventilados, sin evidencia de infiltrados ni neumonía.",
                "keywords": "infiltrado pulmonar (Sin neumonía), neumonía (Sin neumonía)",
                "estado": "Evaluado", "min_atras": 135,
            },
            {
                "rut": "21908412-9", "edad": 28, "sexo": "F", "dias_evolucion": 2,
                "asma": False, "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
                "temperatura": 37.2, "frecuencia_cardiaca": 82, "crepitantes": False, "murmullo_disminuido": False,
                "frecuencia_respiratoria": 16, "sat_o2": 99.0, "matidez": False, "egofonia": False, "soplo_tubarico": False,
                "compromiso_conciencia": False, "rinorrea_odinofagia": True, "antipiretico_6h": False,
                "prob_cnn": 0.09, "prob_nlp": 0.05, "cat_rx": "Sin neumonía", "tipo_doc": "INFORME RADIOLOGICO",
                "texto_med": "Cuadro viral respiratorio alto. Radiografía de tórax normal, se descarta neumonía.",
                "keywords": "neumonía (Sin neumonía)",
                "estado": "Evaluado", "min_atras": 190,
            },
            {
                "rut": "19641230-1", "edad": 69, "sexo": "M", "dias_evolucion": 6,
                "asma": False, "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
                "temperatura": 38.7, "frecuencia_cardiaca": 109, "crepitantes": True, "murmullo_disminuido": True,
                "frecuencia_respiratoria": 28, "sat_o2": 90.0, "matidez": True, "egofonia": True, "soplo_tubarico": False,
                "compromiso_conciencia": False, "rinorrea_odinofagia": False, "antipiretico_6h": False,
                "prob_cnn": 0.91, "prob_nlp": None, "cat_rx": "Sin informe", "tipo_doc": "SIN DOCUMENTO PDF",
                "texto_med": "Evaluación en Modo Triage (Heckerling 5/5). A la espera de informe radiológico formal.",
                "keywords": "",
                "estado": "En evaluación", "min_atras": 32,
            },
        ]

        for c in casos_demo:
            fecha = datetime.now() - timedelta(minutes=c["min_atras"])
            rf_res = rf_predictor(c)
            modo_str = "Antes del informe (Triage)" if c["prob_nlp"] is None else "Con informe (Doble Chequeo)"
            fusion = fusion_fn(
                datos_clinicos=c,
                prob_rf=rf_res["prob_rf"],
                prob_cnn=c["prob_cnn"],
                prob_nlp=c["prob_nlp"],
                modo=modo_str,
            )

            conn.execute(
                pacientes.insert().values(
                    id=c["rut"],
                    edad=c["edad"],
                    sexo=c["sexo"],
                    dias_evolucion=c["dias_evolucion"],
                    asma=c["asma"],
                    epoc=c["epoc"],
                    otra_enf_pulmonar=c["otra_enf_pulmonar"],
                    insuficiencia_cardiaca=c["insuficiencia_cardiaca"],
                    inmunosupresion=c["inmunosupresion"],
                    demencia=c["demencia"],
                    temperatura=c["temperatura"],
                    frecuencia_cardiaca=c["frecuencia_cardiaca"],
                    crepitantes=c["crepitantes"],
                    murmullo_disminuido=c["murmullo_disminuido"],
                    puntaje_heckerling=rf_res["heckerling"]["puntaje"],
                    frecuencia_respiratoria=c["frecuencia_respiratoria"],
                    sat_o2=c["sat_o2"],
                    matidez=c["matidez"],
                    egofonia=c["egofonia"],
                    soplo_tubarico=c["soplo_tubarico"],
                    compromiso_conciencia=c["compromiso_conciencia"],
                    rinorrea_odinofagia=c["rinorrea_odinofagia"],
                    antipiretico_6h=c["antipiretico_6h"],
                    prob_rf=rf_res["prob_rf"],
                    prob_cnn=c["prob_cnn"],
                    prob_nlp=c["prob_nlp"],
                    prob_integrada=fusion["prob_integrada"],
                    categoria_rx_nlp=c["cat_rx"],
                    concordancia_ia=fusion["concordancia"]["estado"],
                    prioridad=fusion["prioridad"],
                    fecha_ingreso=fecha,
                    estado=c["estado"],
                )
            )
            conn.execute(
                imagenes.insert().values(
                    paciente_id=c["rut"],
                    ruta_imagen=None,
                    tipo="Rx actual (caso ejemplo)",
                    prob_ia=c["prob_cnn"],
                    prediccion_ia="Sospecha de neumonía" if c["prob_cnn"] >= 0.5 else "Sin sospecha en imagen",
                    modo_cnn="Caso de ejemplo",
                    fecha=fecha,
                )
            )
            conn.execute(
                diagnosticos.insert().values(
                    paciente_id=c["rut"],
                    tipo_documento=c["tipo_doc"],
                    texto_medico=c["texto_med"],
                    categoria_rx=c["cat_rx"],
                    prob_nlp=c["prob_nlp"],
                    palabras_clave=c["keywords"],
                    diagnostico_final="Pendiente de validación" if c["estado"] != "Evaluado" else "Validado (ejemplo)",
                    concordancia=fusion["concordancia"]["estado"],
                    pdf_path=None,
                    fecha=fecha,
                )
            )
            conn.execute(
                alertas.insert().values(
                    paciente_id=c["rut"],
                    prioridad=fusion["prioridad"],
                    motivo=fusion["motivo"],
                    estado_alerta="Pendiente" if fusion["prioridad"] in ("ROJO", "AMARILLO") else "Informativa",
                    fecha=fecha,
                )
            )
