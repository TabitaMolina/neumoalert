"""
NEUMOALERT — Sistema de Apoyo Clínico para Triage y Sospecha de Neumonía
Diseñado con lo justo y necesario para funcionarios del servicio de salud:
  1. Ingreso y Evaluación (Ficha Heckerling + Radiografía + Informe PDF)
  2. Cola de Urgencias y Alertas
  3. Ficha del Paciente e Informes (Radiólogo / Historial / CSV)

Herramienta de apoyo a la decisión clínica. No sustituye el criterio profesional.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from src.clinical_rf import HECKERLING_REF_TABLE, predict_clinical_rf
from src.cnn_adapter import detect_cnn_backend, predict_cnn
from src.database import (
    BASE_DIR,
    alertas,
    create_db_engine,
    diagnosticos,
    guardar_archivo,
    imagenes,
    insertar_casos_demo_si_vacia,
    obtener_registros,
    pacientes,
    rut_aleatorio,
    rut_dv,
)
from src.multimodal_fusion import fusionar_predicciones
from src.nlp_radiology import analizar_informe_nlp, analizar_pdf_completo

# ============================================================
# CONFIGURACIÓN Y ESTILO CLÍNICO SOBRIO Y RESPONSIVO
# ============================================================
st.set_page_config(
    page_title="NEUMOALERT | Urgencia Respiratoria",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_PDF_DIR = BASE_DIR / "data" / "sample_pdfs"

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .block-container,
    section[data-testid="stMain"] .block-container,
    div[data-testid="stMainBlockContainer"] {
        padding-top: 3.75rem !important;
        padding-bottom: 2rem;
        padding-left: clamp(0.85rem, 2.4vw, 2rem);
        padding-right: clamp(0.85rem, 2.4vw, 2rem);
        max-width: 1360px;
    }

    /* Textos siempre completos sin recortes "..." */
    div[data-testid="stCheckbox"] label,
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stCheckbox"] span,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] p,
    button[data-baseweb="tab"],
    button[data-baseweb="tab"] p,
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button p,
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        overflow-wrap: break-word !important;
        line-height: 1.32 !important;
        height: auto !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        gap: 0.35rem;
    }

    .header-bar {
        background: linear-gradient(135deg, #0b2d48 0%, #0f4c75 100%);
        color: #ffffff;
        padding: 1rem 1.35rem;
        border-radius: 14px;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.6rem;
    }
    .header-title { font-size: clamp(1.25rem, 2.2vw, 1.55rem); font-weight: 800; margin: 0; }
    .header-sub { font-size: 0.86rem; color: #dbeafe; margin: 0.15rem 0 0 0; }

    .verdict-box {
        border-radius: 14px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 0.85rem;
        border-width: 2px;
        border-style: solid;
    }
    .verdict-red { background: #fef2f2; border-color: #ef4444; }
    .verdict-yellow { background: #fffbeb; border-color: #f59e0b; }
    .verdict-green { background: #f0fdf4; border-color: #22c55e; }
    .verdict-neutral { background: #f8fafc; border-color: #cbd5e1; }

    .verdict-kicker { font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; }
    .verdict-main {
        font-size: clamp(1.35rem, 2.5vw, 1.95rem);
        font-weight: 800;
        margin: 0.2rem 0;
        color: #0f172a;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .verdict-text { font-size: 0.9rem; color: #334155; margin-top: 0.3rem; line-height: 1.4; }

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        height: 100%;
    }
    .info-label { font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; }
    .info-val { font-size: clamp(1.05rem, 1.8vw, 1.38rem); font-weight: 800; color: #0f172a; margin: 0.15rem 0; }
    .info-sub { font-size: 0.78rem; color: #475569; }

    .heck-pill {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #0284c7;
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin: 0.45rem 0 0.75rem 0;
        font-size: 0.9rem;
    }

    .badge {
        display: inline-block;
        padding: 0.22rem 0.64rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
    }
    .badge-red { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .badge-yellow { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .badge-green { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-gray { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

    .kw-chip {
        display: inline-block;
        margin: 0.15rem 0.2rem 0.15rem 0;
        padding: 0.2rem 0.52rem;
        border-radius: 7px;
        font-size: 0.76rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 10px 12px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] { font-size: clamp(1.1rem, 2vw, 1.6rem) !important; }

    @media (max-width: 1024px) {
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.65rem !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width: 220px !important; flex: 1 1 220px !important; }
    }
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width: 100% !important; flex: 1 1 100% !important; }
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    return create_db_engine()


engine = get_db()
cnn_status = detect_cnn_backend()

try:
    insertar_casos_demo_si_vacia(engine, predict_clinical_rf, fusionar_predicciones)
except Exception as exc:
    st.error(f"Error al inicializar base de datos o modelos: {exc}")
    st.stop()


# ============================================================
# ESTADO DEL FORMULARIO (LIMPIO POR DEFECTO)
# ============================================================
DEMO_CASES = {
    "grave": {
        "edad": 70, "sexo": "M", "dias_evolucion": 4,
        "temperatura": 39.0, "frecuencia_cardiaca": 114, "crepitantes": True, "murmullo_disminuido": True, "asma": False,
        "sat_o2": 89.0, "frecuencia_respiratoria": 29, "rinorrea_odinofagia": False, "antipiretico_6h": False, "compromiso_conciencia": True,
        "matidez": True, "egofonia": False, "soplo_tubarico": True,
        "epoc": True, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 93, "pdf_sample": "01_rx_definitiva_neumonia.pdf",
    },
    "paracetamol": {
        "edad": 58, "sexo": "F", "dias_evolucion": 5,
        "temperatura": 37.3, "frecuencia_cardiaca": 106, "crepitantes": True, "murmullo_disminuido": False, "asma": False,
        "sat_o2": 92.0, "frecuencia_respiratoria": 25, "rinorrea_odinofagia": False, "antipiretico_6h": True, "compromiso_conciencia": False,
        "matidez": False, "egofonia": False, "soplo_tubarico": False,
        "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 79, "pdf_sample": "02_rx_probable_neumonia.pdf",
    },
    "resfrio": {
        "edad": 30, "sexo": "F", "dias_evolucion": 2,
        "temperatura": 37.1, "frecuencia_cardiaca": 82, "crepitantes": False, "murmullo_disminuido": False, "asma": False,
        "sat_o2": 99.0, "frecuencia_respiratoria": 16, "rinorrea_odinofagia": True, "antipiretico_6h": False, "compromiso_conciencia": False,
        "matidez": False, "egofonia": False, "soplo_tubarico": False,
        "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 12, "pdf_sample": "04_rx_sin_neumonia_con_negaciones.pdf",
    },
    "discrepancia": {
        "edad": 65, "sexo": "F", "dias_evolucion": 5,
        "temperatura": 38.4, "frecuencia_cardiaca": 108, "crepitantes": True, "murmullo_disminuido": True, "asma": False,
        "sat_o2": 91.0, "frecuencia_respiratoria": 26, "rinorrea_odinofagia": False, "antipiretico_6h": False, "compromiso_conciencia": False,
        "matidez": False, "egofonia": False, "soplo_tubarico": False,
        "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 88, "pdf_sample": "04_rx_sin_neumonia_con_negaciones.pdf",
    },
}


def limpiar_formulario_cb() -> None:
    st.session_state["f_rut"] = ""
    st.session_state["f_edad"] = None
    st.session_state["f_sexo"] = None
    st.session_state["f_dias_evolucion"] = None
    st.session_state["f_temperatura"] = None
    st.session_state["f_frecuencia_cardiaca"] = None
    st.session_state["f_sat_o2"] = None
    st.session_state["f_frecuencia_respiratoria"] = None
    for bkey in [
        "f_crepitantes", "f_murmullo_disminuido", "f_asma", "f_rinorrea_odinofagia",
        "f_antipiretico_6h", "f_compromiso_conciencia", "f_matidez", "f_egofonia",
        "f_soplo_tubarico", "f_epoc", "f_otra_enf_pulmonar", "f_insuficiencia_cardiaca",
        "f_inmunosupresion", "f_demencia", "f_usar_cnn_demo",
    ]:
        st.session_state[bkey] = False
    st.session_state["f_cnn_pct"] = 0
    st.session_state["f_pdf_sample"] = None
    st.session_state["f_texto_informe"] = ""
    st.session_state["f_nonce"] = st.session_state.get("f_nonce", 0) + 1


def cargar_ejemplo_cb(key: str) -> None:
    c = DEMO_CASES[key]
    st.session_state["f_rut"] = rut_aleatorio()
    for k, v in c.items():
        if k not in ("cnn_pct", "pdf_sample"):
            st.session_state[f"f_{k}"] = v
    st.session_state["f_usar_cnn_demo"] = not cnn_status["available"]
    st.session_state["f_cnn_pct"] = c["cnn_pct"]
    st.session_state["f_pdf_sample"] = c["pdf_sample"]
    st.session_state["f_texto_informe"] = ""
    st.session_state["f_nonce"] = st.session_state.get("f_nonce", 0) + 1


def init_state() -> None:
    if "f_rut" not in st.session_state:
        limpiar_formulario_cb()
    elif st.session_state.pop("limpiar_tras_guardar", False):
        limpiar_formulario_cb()


init_state()


def priority_badge(prio: str) -> str:
    cls = {"ROJO": "badge-red", "AMARILLO": "badge-yellow", "VERDE": "badge-green"}.get(prio, "badge-gray")
    txt = {"ROJO": "🔴 PRIORIDAD ALTA", "AMARILLO": "🟡 PRIORIDAD MEDIA", "VERDE": "🟢 BAJA PRIORIDAD"}.get(prio, prio)
    return f'<span class="badge {cls}">{txt}</span>'


def render_keyword_chips(palabras_clave: list[dict]) -> None:
    if not palabras_clave:
        return
    styles = {
        "Definitiva": "background:#fee2e2;color:#b91c1c;border:1px solid #fecaca;",
        "Probable": "background:#ffedd5;color:#c2410c;border:1px solid #fed7aa;",
        "Posible": "background:#fef3c7;color:#b45309;border:1px solid #fde68a;",
        "Sin neumonía": "background:#dcfce7;color:#15803d;border:1px solid #bbf7d0;",
        "Contexto": "background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;",
    }
    html = "".join(
        f'<span class="kw-chip" style="{styles.get(k["polaridad"], styles["Contexto"])}">'
        f'{k["termino"]} ({k["polaridad"]})</span>'
        for k in palabras_clave
    )
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# NAVEGACIÓN LATERAL (3 MÓDULOS CLÍNICOS ESENCIALES)
# ============================================================
st.sidebar.markdown("## 🫁 NEUMOALERT")
st.sidebar.caption("Urgencia Respiratoria")

pagina = st.sidebar.radio(
    "Módulo",
    [
        "🩺 Ingreso y Evaluación",
        "🚨 Cola de Urgencias",
        "🩻 Ficha e Informes",
    ],
)

st.sidebar.divider()
with st.sidebar.expander("⚙️ Modo prueba", expanded=False):
    st.caption("Solo para pruebas o demostraciones:")
    st.button("🔴 Caso neumonía grave", width="stretch", on_click=cargar_ejemplo_cb, args=("grave",))
    st.button("🟡 Caso con paracetamol", width="stretch", on_click=cargar_ejemplo_cb, args=("paracetamol",))
    st.button("🟢 Caso resfrío común", width="stretch", on_click=cargar_ejemplo_cb, args=("resfrio",))
    st.button("⚡ Caso discrepancia", width="stretch", on_click=cargar_ejemplo_cb, args=("discrepancia",))
    if not cnn_status["available"]:
        st.checkbox("Ingresar % de imagen manualmente", key="f_usar_cnn_demo")
        if st.session_state.get("f_usar_cnn_demo"):
            st.slider("Sospecha en radiografía (%)", 0, 100, key="f_cnn_pct")

st.sidebar.caption("Herramienta de apoyo clínico. No sustituye evaluación médica.")

# ============================================================
# ENCABEZADO SIMPLE
# ============================================================
st.markdown(
    """
<div class="header-bar">
    <div>
        <p class="header-title">🫁 NEUMOALERT</p>
        <p class="header-sub">Apoyo al triage y evaluación de sospecha de neumonía</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 1. INGRESO Y EVALUACIÓN CLÍNICA
# ============================================================
if pagina == "🩺 Ingreso y Evaluación":
    col_in, col_out = st.columns([1.22, 1.0], gap="large")
    nonce = st.session_state.get("f_nonce", 0)

    with col_in:
        h_top1, h_top2 = st.columns([2.2, 1.0], vertical_alignment="center")
        h_top1.markdown("#### Datos del paciente")
        h_top2.button("🧹 Limpiar ficha", width="stretch", on_click=limpiar_formulario_cb)

        tab_ficha, tab_adj = st.tabs(["📋 Ficha Clínica", "🩻 Radiografía e Informe"])

        with tab_ficha:
            id1, id2, id3 = st.columns([1.6, 0.85, 0.85])
            rut_val = id1.text_input("RUT", key="f_rut", placeholder="Ej. 16482910-3")
            edad = id2.number_input("Edad", min_value=18, max_value=105, step=1, placeholder="Ej. 54", key="f_edad")
            sexo = id3.selectbox("Sexo", ["F", "M"], placeholder="Elegir...", key="f_sexo")

            st.markdown("**Signos vitales (primer control)**")
            v1, v2, v3, v4 = st.columns(4)
            temperatura = v1.number_input("Temp. (°C)", min_value=35.0, max_value=42.0, step=0.1, placeholder="37.5", key="f_temperatura", help="Axilar. Suma punto Heckerling si > 37,8 °C")
            frecuencia_cardiaca = v2.number_input("FC (lpm)", min_value=40, max_value=180, step=1, placeholder="90", key="f_frecuencia_cardiaca", help="En reposo. Suma punto Heckerling si > 100 lpm")
            sat_o2 = v3.number_input("SatO₂ (%)", min_value=60.0, max_value=100.0, step=1.0, placeholder="96", key="f_sat_o2", help="Sin oxígeno suplementario")
            fr = v4.number_input("FR (rpm)", min_value=8, max_value=55, step=1, placeholder="20", key="f_frecuencia_respiratoria")

            st.markdown("**Examen físico y síntomas clave**")
            e1, e2, e3 = st.columns(3)
            with e1:
                crepitantes = st.checkbox("Crepitaciones", key="f_crepitantes", help="Ruidos cortos inspiratorios que no desaparecen al toser")
                asma = st.checkbox("Tiene asma", key="f_asma", help="En Heckerling, la ausencia de asma suma 1 punto")
            with e2:
                murmullo_disminuido = st.checkbox("Murmullo disminuido", key="f_murmullo_disminuido", help="Disminución localizada respecto al lado opuesto")
                rinorrea = st.checkbox("Rinorrea / odinofagia", key="f_rinorrea_odinofagia", help="Sugiere cuadro viral alto (resfrío)")
            with e3:
                antipiretico = st.checkbox("Antipirético ≤ 6 h", key="f_antipiretico_6h", help="Puede ocultar fiebre en el primer control")
                conciencia = st.checkbox("Compromiso conciencia", key="f_compromiso_conciencia", help="Glasgow < 15 o desorientación nueva")

            d_col, h_col = st.columns([0.85, 1.65], vertical_alignment="bottom")
            dias_evol = d_col.number_input("Días evolución", min_value=1, max_value=14, step=1, placeholder="3", key="f_dias_evolucion")
            with h_col:
                if temperatura is not None and frecuencia_cardiaca is not None:
                    pt_h = (
                        int(temperatura > 37.8)
                        + int(frecuencia_cardiaca > 100)
                        + int(bool(crepitantes))
                        + int(bool(murmullo_disminuido))
                        + int(not bool(asma))
                    )
                    ref_h = HECKERLING_REF_TABLE[pt_h]
                    st.markdown(f'<div class="heck-pill"><b>Puntaje Heckerling: {pt_h} / 5</b> · {ref_h["zona"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="heck-pill" style="background:#f8fafc;border-color:#cbd5e1;border-left-color:#94a3b8;"><b>Puntaje Heckerling:</b> pendiente</div>', unsafe_allow_html=True)

            with st.expander("➕ Antecedentes médicos y signos focales (opcional)", expanded=False):
                b1, b2 = st.columns(2)
                with b1:
                    epoc = st.checkbox("EPOC", key="f_epoc")
                    otra_pulm = st.checkbox("Otra enf. pulmonar", key="f_otra_enf_pulmonar")
                    icc = st.checkbox("Insuf. cardíaca (ICC)", key="f_insuficiencia_cardiaca")
                    inmuno = st.checkbox("Inmunosupresión", key="f_inmunosupresion")
                    demencia = st.checkbox("Demencia", key="f_demencia")
                with b2:
                    matidez = st.checkbox("Matidez", key="f_matidez")
                    egofonia = st.checkbox("Egofonía", key="f_egofonia")
                    soplo = st.checkbox("Soplo tubárico", key="f_soplo_tubarico")

        with tab_adj:
            rx_file = st.file_uploader(
                "Radiografía de tórax actual (JPG, PNG o DICOM)",
                type=["jpg", "jpeg", "png", "dcm"],
                key=f"rx_up_{nonce}",
            )
            if rx_file is not None:
                if rx_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    st.image(rx_file, caption=rx_file.name, width="stretch")
                else:
                    st.info(f"Archivo DICOM cargado: {rx_file.name}")

            pdf_up = st.file_uploader(
                "Informe radiológico en PDF (opcional)",
                type=["pdf"],
                key=f"pdf_up_{nonce}",
            )

            with st.expander("➕ Radiografía anterior o texto de informe", expanded=False):
                rx_antigua = st.file_uploader("Radiografía antigua (comparación)", type=["jpg", "jpeg", "png", "dcm"], key=f"rx_ant_{nonce}")
                texto_informe = st.text_area("Pegar texto del informe (si no tiene PDF)", height=75, key="f_texto_informe")

    # ---------------- EVALUACIÓN EN TIEMPO REAL ----------------
    ficha_completa = all(v is not None for v in (edad, sexo, temperatura, frecuencia_cardiaca, sat_o2, fr, dias_evol))

    if ficha_completa:
        datos_paciente = {
            "edad": int(edad), "sexo": str(sexo), "dias_evolucion": int(dias_evol),
            "asma": bool(asma), "epoc": bool(epoc), "otra_enf_pulmonar": bool(otra_pulm),
            "insuficiencia_cardiaca": bool(icc), "inmunosupresion": bool(inmuno), "demencia": bool(demencia),
            "temperatura": float(temperatura), "frecuencia_cardiaca": int(frecuencia_cardiaca),
            "crepitantes": bool(crepitantes), "murmullo_disminuido": bool(murmullo_disminuido),
            "frecuencia_respiratoria": int(fr), "sat_o2": float(sat_o2),
            "matidez": bool(matidez), "egofonia": bool(egofonia), "soplo_tubarico": bool(soplo),
            "compromiso_conciencia": bool(conciencia), "rinorrea_odinofagia": bool(rinorrea),
            "antipiretico_6h": bool(antipiretico),
        }
        rf_res = predict_clinical_rf(datos_paciente)
    else:
        datos_paciente = {}
        rf_res = None

    rx_bytes = rx_file.getvalue() if (rx_file is not None and not rx_file.name.lower().endswith(".dcm")) else None
    cnn_manual = (st.session_state.get("f_cnn_pct", 0) / 100.0) if st.session_state.get("f_usar_cnn_demo") else None
    cnn_res = predict_cnn(image_source=rx_bytes, manual_cnn_prob=cnn_manual)

    # PDF subido por funcionario (o PDF de caso de prueba si activó Modo prueba)
    pdf_bytes = None
    pdf_name = None
    if pdf_up is not None:
        pdf_bytes = pdf_up.getvalue()
        pdf_name = pdf_up.name
    elif st.session_state.get("f_pdf_sample"):
        sample_p = SAMPLE_PDF_DIR / st.session_state["f_pdf_sample"]
        if sample_p.exists():
            pdf_bytes = sample_p.read_bytes()
            pdf_name = sample_p.name

    if pdf_bytes is not None:
        nlp_res = analizar_pdf_completo(pdf_bytes)
    elif texto_informe.strip():
        nlp_res = analizar_informe_nlp(texto_informe.strip())
    else:
        nlp_res = {
            "es_valido_paciente": False,
            "es_antecedente_previo": False,
            "tipo_documento": "SIN DOCUMENTO PDF",
            "prob_nlp": None,
            "clase_nlp": "Sin informe",
            "hallazgos": [],
            "palabras_clave": [],
            "resumen": "",
        }

    hay_datos = (rf_res is not None) or (cnn_res["prob_cnn"] is not None) or bool(nlp_res.get("es_valido_paciente"))

    # ---------------- PANEL DERECHO: RESUMEN CLÍNICO ----------------
    with col_out:
        st.markdown("#### Evaluación de apoyo")

        if not hay_datos:
            st.markdown(
                """
            <div class="verdict-box verdict-neutral">
                <div class="verdict-kicker">En espera de datos</div>
                <div class="verdict-main" style="font-size:1.35rem;color:#334155;">Ficha lista para ingreso</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            tiene_informe_actual = bool(nlp_res.get("es_valido_paciente")) and not bool(nlp_res.get("es_antecedente_previo"))
            fusion = fusionar_predicciones(
                datos_clinicos=datos_paciente,
                prob_rf=rf_res["prob_rf"] if rf_res is not None else None,
                prob_cnn=cnn_res["prob_cnn"],
                prob_nlp=nlp_res["prob_nlp"] if tiene_informe_actual else None,
                modo="Con informe" if tiene_informe_actual else "Triage",
                es_antecedente_previo=bool(nlp_res.get("es_antecedente_previo", False)),
            )

            v_cls = {"ROJO": "verdict-red", "AMARILLO": "verdict-yellow", "VERDE": "verdict-green"}[fusion["prioridad"]]
            st.markdown(
                f"""
            <div class="verdict-box {v_cls}">
                <div class="verdict-kicker">Sospecha integrada de neumonía</div>
                <div class="verdict-main">{fusion['prob_integrada']:.0%} {priority_badge(fusion['prioridad'])}</div>
                <div class="verdict-text">{fusion['accion_sugerida']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            conc = fusion["concordancia"]
            if conc["color"] == "rojo":
                st.error(f"⚠️ **{conc['estado']}:** {conc['mensaje']}")
            elif "DISCREPANCIA" in conc["estado"]:
                st.warning(f"🔎 **{conc['estado']}:** {conc['mensaje']}")

            c1, c2, c3 = st.columns(3)
            with c1:
                rf_txt = f"{rf_res['prob_rf']:.0%}" if rf_res is not None else "—"
                rf_sub = f"Heckerling {rf_res['heckerling']['puntaje']}/5" if rf_res is not None else "Pendiente"
                st.markdown(
                    f'<div class="info-card"><div class="info-label">Clínica</div><div class="info-val">{rf_txt}</div><div class="info-sub">{rf_sub}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                rx_txt = f"{cnn_res['prob_cnn']:.0%}" if cnn_res["prob_cnn"] is not None else "—"
                rx_sub = "Evaluada" if cnn_res["prob_cnn"] is not None else "Sin imagen"
                st.markdown(
                    f'<div class="info-card"><div class="info-label">Radiografía</div><div class="info-val">{rx_txt}</div><div class="info-sub">{rx_sub}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="info-card"><div class="info-label">Informe</div><div class="info-val" style="font-size:1.05rem;">{nlp_res["clase_nlp"]}</div><div class="info-sub">{pdf_name or "No adjunto"}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

            # Factores clínicos concisos
            if rf_res is not None:
                exp = rf_res["explicacion"]
                if exp["a_favor"]:
                    st.markdown("**Hallazgos de alerta:** " + " · ".join(exp["a_favor"][:4]))
                if exp["en_contra"]:
                    st.markdown("**A favor de otra causa:** " + " · ".join(exp["en_contra"][:2]))
                for nota in exp["notas"]:
                    st.info(nota)

            if nlp_res.get("tipo_documento") == "PROTOCOLO O GUÍA CLÍNICA":
                st.warning("El PDF adjunto corresponde a una guía clínica general, no a un informe de paciente.")
            elif nlp_res.get("palabras_clave"):
                st.markdown("**Hallazgos en el informe:**")
                render_keyword_chips(nlp_res["palabras_clave"])

            if nlp_res.get("hallazgos"):
                with st.expander("Ver extracto del informe PDF", expanded=False):
                    for h in nlp_res["hallazgos"][:6]:
                        st.markdown(f"- **[{h['estado']}]** {h['oracion']}")

            st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
            if st.button("💾 Registrar paciente", type="primary", width="stretch"):
                rut_clean = rut_val.strip().upper().replace(".", "")
                if not ficha_completa:
                    st.warning("Completa los signos vitales, edad, sexo y días de evolución antes de registrar.")
                elif not re.fullmatch(r"\d{7,8}-[\dK]", rut_clean):
                    st.error("Ingresa un RUT válido (ej. 16482910-3).")
                else:
                    r_num, dv = rut_clean.split("-")
                    if rut_dv(r_num) != dv:
                        st.error("El dígito verificador del RUT no es válido.")
                    else:
                        try:
                            ruta_rx = guardar_archivo(rx_file, rut_clean, "rx_actual")
                            ruta_rx_ant = guardar_archivo(rx_antigua, rut_clean, "rx_antigua")
                            ruta_pdf = guardar_archivo(pdf_up, rut_clean, "informe_pdf") if pdf_up else (
                                str(SAMPLE_PDF_DIR / pdf_name) if (pdf_name and st.session_state.get("f_pdf_sample")) else None
                            )
                            fecha = datetime.now()
                            kw_str = ", ".join(f"{k['termino']} ({k['polaridad']})" for k in nlp_res.get("palabras_clave", []))
                            txt_doc = texto_informe.strip() or (
                                " | ".join(h["oracion"] for h in nlp_res["hallazgos"][:3]) if nlp_res.get("hallazgos") else nlp_res.get("resumen", "")
                            )

                            with engine.begin() as conn:
                                conn.execute(
                                    pacientes.insert().values(
                                        id=rut_clean,
                                        puntaje_heckerling=rf_res["heckerling"]["puntaje"],
                                        prob_rf=rf_res["prob_rf"],
                                        prob_cnn=cnn_res["prob_cnn"],
                                        prob_nlp=nlp_res.get("prob_nlp"),
                                        prob_integrada=fusion["prob_integrada"],
                                        categoria_rx_nlp=nlp_res.get("clase_nlp", "Sin informe"),
                                        concordancia_ia=fusion["concordancia"]["estado"],
                                        prioridad=fusion["prioridad"],
                                        fecha_ingreso=fecha,
                                        estado="En evaluación",
                                        **datos_paciente,
                                    )
                                )
                                conn.execute(
                                    imagenes.insert().values(
                                        paciente_id=rut_clean,
                                        ruta_imagen=ruta_rx,
                                        tipo="Rx actual",
                                        prob_ia=cnn_res["prob_cnn"],
                                        prediccion_ia=cnn_res["prediccion_cnn"],
                                        modo_cnn=cnn_res["modo"],
                                        fecha=fecha,
                                    )
                                )
                                if ruta_rx_ant:
                                    conn.execute(
                                        imagenes.insert().values(
                                            paciente_id=rut_clean,
                                            ruta_imagen=ruta_rx_ant,
                                            tipo="Rx antigua",
                                            prob_ia=None,
                                            prediccion_ia="Referencia previa",
                                            modo_cnn="HISTORIAL",
                                            fecha=fecha,
                                        )
                                    )
                                conn.execute(
                                    diagnosticos.insert().values(
                                        paciente_id=rut_clean,
                                        tipo_documento=nlp_res.get("tipo_documento", "SIN DOCUMENTO PDF"),
                                        texto_medico=txt_doc,
                                        categoria_rx=nlp_res.get("clase_nlp", "Sin informe"),
                                        prob_nlp=nlp_res.get("prob_nlp"),
                                        palabras_clave=kw_str,
                                        diagnostico_final="Pendiente de validación",
                                        concordancia=fusion["concordancia"]["estado"],
                                        pdf_path=ruta_pdf,
                                        fecha=fecha,
                                    )
                                )
                                conn.execute(
                                    alertas.insert().values(
                                        paciente_id=rut_clean,
                                        prioridad=fusion["prioridad"],
                                        motivo=fusion["motivo"],
                                        estado_alerta="Pendiente" if fusion["prioridad"] in ("ROJO", "AMARILLO") else "Informativa",
                                        fecha=fecha,
                                    )
                                )
                            st.success(f"Paciente **{rut_clean}** registrado correctamente.")
                            st.session_state["limpiar_tras_guardar"] = True
                        except SQLAlchemyError as exc:
                            st.error(f"No se pudo registrar (el RUT ya existe en el sistema). Detalle: {exc}")


# ============================================================
# 2. COLA DE URGENCIAS Y ALERTAS
# ============================================================
elif pagina == "🚨 Cola de Urgencias":
    df_p = obtener_registros(engine, pacientes)
    df_a = obtener_registros(engine, alertas)

    if df_p.empty:
        st.info("No hay pacientes en espera.")
    else:
        orden = {"ROJO": 0, "AMARILLO": 1, "VERDE": 2}
        df_p["orden"] = df_p["prioridad"].map(orden).fillna(3)
        df_p = df_p.sort_values(["orden", "prob_integrada", "fecha_ingreso"], ascending=[True, False, True])

        rojos = int((df_p["prioridad"] == "ROJO").sum())
        amarillos = int((df_p["prioridad"] == "AMARILLO").sum())
        verdes = int((df_p["prioridad"] == "VERDE").sum())
        discs = int(df_p["concordancia_ia"].fillna("").str.contains("DISCREPANCIA", case=False).sum())

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🔴 Urgente", rojos)
        k2.metric("🟡 Preferente", amarillos)
        k3.metric("🟢 Baja prioridad", verdes)
        k4.metric("⚠️ Discrepancias", discs)

        t_cola, t_al = st.tabs(["Pacientes en espera", "Alertas pendientes"])

        with t_cola:
            f1, f2, f3 = st.columns([1.2, 1.0, 1.0], vertical_alignment="bottom")
            filtro_p = f1.multiselect("Prioridad", ["ROJO", "AMARILLO", "VERDE"], default=["ROJO", "AMARILLO", "VERDE"])
            solo_disc = f2.checkbox("Solo discrepancias")
            buscar_r = f3.text_input("Buscar RUT")

            vista = df_p[df_p["prioridad"].isin(filtro_p)].copy()
            if solo_disc:
                vista = vista[vista["concordancia_ia"].fillna("").str.contains("DISCREPANCIA", case=False)]
            if buscar_r:
                vista = vista[vista["id"].str.contains(buscar_r.strip(), case=False, na=False)]

            vf = vista[["prioridad", "id", "edad", "puntaje_heckerling", "sat_o2", "temperatura", "prob_integrada", "categoria_rx_nlp", "estado"]].copy()
            vf["puntaje_heckerling"] = vf["puntaje_heckerling"].apply(lambda x: f"{int(x)}/5" if pd.notna(x) else "—")
            vf["prob_integrada"] = vf["prob_integrada"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
            vf.columns = ["Prioridad", "RUT", "Edad", "Heckerling", "SatO₂", "Temp.", "Sospecha", "Informe", "Estado"]
            st.dataframe(vf, width="stretch", hide_index=True)

        with t_al:
            pend = df_a[df_a["estado_alerta"].isin(["Pendiente", "Enviada"])].sort_values("fecha", ascending=False)
            if pend.empty:
                st.success("Sin alertas pendientes.")
            else:
                st.dataframe(
                    pend[["id", "prioridad", "paciente_id", "motivo", "estado_alerta", "fecha"]].rename(
                        columns={"id": "ID", "prioridad": "Prioridad", "paciente_id": "RUT", "motivo": "Motivo", "estado_alerta": "Estado", "fecha": "Fecha"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                u1, u2, u3 = st.columns([1.0, 1.2, 1.0], vertical_alignment="bottom")
                aid = u1.selectbox("Alerta (ID)", pend["id"].tolist())
                nest = u2.selectbox("Marcar como", ["Revisada", "Cerrada", "Enviada"])
                if u3.button("Actualizar", type="primary", width="stretch"):
                    with engine.begin() as conn:
                        conn.execute(alertas.update().where(alertas.c.id == int(aid)).values(estado_alerta=nest))
                    st.rerun()


# ============================================================
# 3. FICHA DEL PACIENTE E INFORMES (RADIÓLOGO / HISTORIAL)
# ============================================================
elif pagina == "🩻 Ficha e Informes":
    df_p = obtener_registros(engine, pacientes)

    if df_p.empty:
        st.info("No hay pacientes registrados.")
    else:
        s1, s2 = st.columns([1.6, 1.0], vertical_alignment="bottom")
        rut_sel = s1.selectbox("Seleccionar paciente por RUT", df_p["id"].tolist())
        csv_data = df_p.to_csv(index=False).encode("utf-8-sig")
        s2.download_button("⬇️ Exportar registro CSV", csv_data, "neumoalert_registro.csv", "text/csv", width="stretch")

        p = df_p[df_p["id"] == rut_sel].iloc[0]
        st.markdown(f"#### RUT `{rut_sel}` &nbsp; {priority_badge(str(p['prioridad']))}", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Heckerling", f"{int(p['puntaje_heckerling'])} / 5")
        m2.metric("SatO₂ / Temp.", f"{p['sat_o2']:.0f}% · {p['temperatura']:.1f} °C")
        m3.metric("Informe actual", str(p["categoria_rx_nlp"] or "Sin informe"))

        with engine.connect() as conn:
            imgs = pd.read_sql(select(imagenes).where(imagenes.c.paciente_id == rut_sel).order_by(imagenes.c.fecha.desc()), conn)
            diags = pd.read_sql(select(diagnosticos).where(diagnosticos.c.paciente_id == rut_sel).order_by(diagnosticos.c.fecha.desc()), conn)

        c_img, c_rad = st.columns([1.0, 1.2], gap="large")

        with c_img:
            st.markdown("**Radiografías**")
            for _, img in imgs.iterrows():
                path = img["ruta_imagen"]
                if path and Path(path).exists() and Path(path).suffix.lower() in (".jpg", ".jpeg", ".png"):
                    st.image(path, caption=img["tipo"], width="stretch")
                else:
                    st.caption(f"{img['tipo']} · Sin archivo de imagen adjunto.")

        with c_rad:
            st.markdown("**Registrar informe radiológico**")
            pdf_rad = st.file_uploader("Adjuntar informe PDF", type=["pdf"], key="rad_pdf")
            txt_rad = st.text_area("Hallazgos / conclusión", height=80)

            nlp_prev = None
            if pdf_rad is not None:
                nlp_prev = analizar_pdf_completo(pdf_rad.getvalue())
            elif txt_rad.strip():
                nlp_prev = analizar_informe_nlp(txt_rad.strip())

            if nlp_prev is not None and nlp_prev.get("es_valido_paciente"):
                st.info(f"Categoría sugerida por lectura del informe: **{nlp_prev['clase_nlp']}**")
                render_keyword_chips(nlp_prev.get("palabras_clave", []))

            with st.form("form_rad_clinico"):
                r1, r2 = st.columns(2)
                cat_med = r1.selectbox("Categoría radiológica", ["Sin neumonía", "Posible", "Probable", "Definitiva"])
                est_med = r2.selectbox("Estado", ["Evaluado", "Pendiente de informe", "En evaluación"])
                diag_med = st.text_input("Conclusión del profesional")
                if st.form_submit_button("💾 Guardar informe", type="primary", width="stretch"):
                    ruta_p = guardar_archivo(pdf_rad, rut_sel, "informe_radiologo") if pdf_rad else None
                    prob_n = (
                        nlp_prev.get("prob_nlp")
                        if (nlp_prev and nlp_prev.get("es_valido_paciente"))
                        else {"Sin neumonía": 0.06, "Posible": 0.45, "Probable": 0.78, "Definitiva": 0.95}[cat_med]
                    )
                    kw_n = ", ".join(f"{k['termino']} ({k['polaridad']})" for k in (nlp_prev.get("palabras_clave", []) if nlp_prev else []))
                    fus_n = fusionar_predicciones(
                        datos_clinicos=p.to_dict(),
                        prob_rf=float(p["prob_rf"]) if pd.notna(p["prob_rf"]) else None,
                        prob_cnn=float(p["prob_cnn"]) if pd.notna(p["prob_cnn"]) else None,
                        prob_nlp=prob_n,
                        modo="Con informe",
                    )
                    with engine.begin() as conn:
                        conn.execute(
                            diagnosticos.insert().values(
                                paciente_id=rut_sel,
                                tipo_documento=nlp_prev.get("tipo_documento", "INFORME RADIOLOGICO") if nlp_prev else "INFORME RADIOLOGICO",
                                texto_medico=txt_rad or f"Clasificado como {cat_med}.",
                                categoria_rx=cat_med,
                                prob_nlp=prob_n,
                                palabras_clave=kw_n,
                                diagnostico_final=diag_med or cat_med,
                                concordancia=fus_n["concordancia"]["estado"],
                                pdf_path=ruta_p,
                                fecha=datetime.now(),
                            )
                        )
                        conn.execute(
                            pacientes.update()
                            .where(pacientes.c.id == rut_sel)
                            .values(
                                prob_nlp=prob_n,
                                categoria_rx_nlp=cat_med,
                                prob_integrada=fus_n["prob_integrada"],
                                prioridad=fus_n["prioridad"],
                                concordancia_ia=fus_n["concordancia"]["estado"],
                                estado=est_med,
                            )
                        )
                    st.success("Informe guardado.")
                    st.rerun()

        if not diags.empty:
            st.markdown("**Historial de informes**")
            st.dataframe(
                diags[["fecha", "categoria_rx", "diagnostico_final", "concordancia"]].rename(
                    columns={"fecha": "Fecha", "categoria_rx": "Categoría", "diagnostico_final": "Conclusión", "concordancia": "Concordancia"}
                ),
                width="stretch",
                hide_index=True,
            )
