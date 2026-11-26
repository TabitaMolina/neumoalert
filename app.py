"""
NEUMOALERT v3 — Interfaz Clínica Intuitiva (Ficha Heckerling + Random Forest + NLP 4 Categorías + CNN)

Diseñada para ser clara en 5 segundos para médicos, estudiantes y jurado:
- 4 pantallas simples en lugar de menús recargados
- 4 botones de "Caso ejemplo con 1 clic" para probar y demostrar al instante
- Vista dividida: formulario guiado a la izquierda y veredicto en tiempo real a la derecha
- Ficha clínica alineada al cuestionario del médico (Heckerling 0–5 + datos adicionales)
- Lectura de informes PDF en las 4 categorías del estudio (Sin neumonía, Posible, Probable, Definitiva)

Prototipo académico demostrativo (Capstone SIC 2026). No sustituye la evaluación médica real.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# CONFIGURACIÓN Y ESTILO LIMPIO
# ============================================================
st.set_page_config(
    page_title="NEUMOALERT | Apoyo Clínico Neumonía",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"
SAMPLE_PDF_DIR = BASE_DIR / "data" / "sample_pdfs"

SAMPLE_PDF_LABELS = {
    "01_rx_definitiva_neumonia.pdf": "🔴 Definitiva — Neumonía confirmada",
    "02_rx_probable_neumonia.pdf": "🟠 Probable — Lo más probable es neumonía",
    "03_rx_posible_dudoso.pdf": "🟡 Posible — Hallazgo dudoso / no se descarta",
    "04_rx_sin_neumonia_con_negaciones.pdf": "🟢 Sin neumonía — Normal (con negaciones)",
    "05_antecedente_epicrisis_previa.pdf": "📁 Epicrisis previa — Antecedente antiguo",
}

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1420px; }

    .top-bar {
        background: linear-gradient(135deg, #0b2d48 0%, #0f4c75 60%, #0e7490 100%);
        color: #ffffff;
        padding: 1.15rem 1.6rem;
        border-radius: 16px;
        margin-bottom: 1.1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
        box-shadow: 0 6px 20px rgba(11, 45, 72, 0.13);
    }
    .top-title { font-size: 1.65rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
    .top-sub { font-size: 0.9rem; color: #dbeafe; margin: 0.2rem 0 0 0; }
    .top-tag {
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .verdict-box {
        border-radius: 16px;
        padding: 1.2rem 1.35rem;
        margin-bottom: 0.9rem;
        border-width: 2px;
        border-style: solid;
    }
    .verdict-red { background: #fef2f2; border-color: #ef4444; }
    .verdict-yellow { background: #fffbeb; border-color: #f59e0b; }
    .verdict-green { background: #f0fdf4; border-color: #22c55e; }

    .verdict-kicker { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #475569; }
    .verdict-main { font-size: 2.15rem; font-weight: 800; margin: 0.15rem 0; line-height: 1.1; color: #0f172a; }
    .verdict-text { font-size: 0.92rem; color: #334155; margin-top: 0.35rem; }

    .mini-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        height: 100%;
    }
    .mini-title { font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
    .mini-num { font-size: 1.55rem; font-weight: 800; color: #0f172a; margin: 0.15rem 0; }
    .mini-sub { font-size: 0.8rem; color: #475569; line-height: 1.3; }

    .heck-banner {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 5px solid #0284c7;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0 0.85rem 0;
    }

    .badge {
        display: inline-block;
        padding: 0.24rem 0.68rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .badge-red { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .badge-yellow { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .badge-green { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-blue { background: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .badge-gray { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

    .kw-chip {
        display: inline-block;
        margin: 0.18rem 0.22rem 0.18rem 0;
        padding: 0.22rem 0.56rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 12px 14px;
        border-radius: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    return create_db_engine()


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


engine = get_db()
metrics_data = load_metrics()
cnn_status = detect_cnn_backend()

try:
    insertar_casos_demo_si_vacia(engine, predict_clinical_rf, fusionar_predicciones)
except Exception as exc:
    st.error(f"Error al inicializar base de datos o modelos: {exc}")
    st.stop()


# ============================================================
# CASOS DE PRUEBA RÁPIDA (1 CLIC)
# ============================================================
PRESETS = {
    "grave": {
        "edad": 70, "sexo": "M", "dias_evolucion": 4,
        "temperatura": 39.0, "frecuencia_cardiaca": 114, "crepitantes": True, "murmullo_disminuido": True, "asma": False,
        "sat_o2": 89.0, "frecuencia_respiratoria": 29, "rinorrea_odinofagia": False, "antipiretico_6h": False, "compromiso_conciencia": True,
        "matidez": True, "egofonia": False, "soplo_tubarico": True,
        "epoc": True, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 93, "modo": "Con informe del radiólogo (Doble Chequeo)",
        "pdf_file": "01_rx_definitiva_neumonia.pdf",
    },
    "paracetamol": {
        "edad": 58, "sexo": "F", "dias_evolucion": 5,
        "temperatura": 37.3, "frecuencia_cardiaca": 106, "crepitantes": True, "murmullo_disminuido": False, "asma": False,
        "sat_o2": 92.0, "frecuencia_respiratoria": 25, "rinorrea_odinofagia": False, "antipiretico_6h": True, "compromiso_conciencia": False,
        "matidez": False, "egofonia": False, "soplo_tubarico": False,
        "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 79, "modo": "Con informe del radiólogo (Doble Chequeo)",
        "pdf_file": "02_rx_probable_neumonia.pdf",
    },
    "resfrio": {
        "edad": 30, "sexo": "F", "dias_evolucion": 2,
        "temperatura": 37.1, "frecuencia_cardiaca": 82, "crepitantes": False, "murmullo_disminuido": False, "asma": False,
        "sat_o2": 99.0, "frecuencia_respiratoria": 16, "rinorrea_odinofagia": True, "antipiretico_6h": False, "compromiso_conciencia": False,
        "matidez": False, "egofonia": False, "soplo_tubarico": False,
        "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 12, "modo": "Con informe del radiólogo (Doble Chequeo)",
        "pdf_file": "04_rx_sin_neumonia_con_negaciones.pdf",
    },
    "discrepancia": {
        "edad": 65, "sexo": "F", "dias_evolucion": 5,
        "temperatura": 38.4, "frecuencia_cardiaca": 108, "crepitantes": True, "murmullo_disminuido": True, "asma": False,
        "sat_o2": 91.0, "frecuencia_respiratoria": 26, "rinorrea_odinofagia": False, "antipiretico_6h": False, "compromiso_conciencia": False,
        "matidez": False, "egofonia": False, "soplo_tubarico": False,
        "epoc": False, "otra_enf_pulmonar": False, "insuficiencia_cardiaca": False, "inmunosupresion": False, "demencia": False,
        "cnn_pct": 88, "modo": "Con informe del radiólogo (Doble Chequeo)",
        "pdf_file": "04_rx_sin_neumonia_con_negaciones.pdf",
    },
}


def generar_nuevo_rut_cb() -> None:
    st.session_state["f_rut"] = rut_aleatorio()


def limpiar_formulario_cb() -> None:
    """Deja todas las casillas completamente en blanco, borra Rx/PDF y reinicia el panel."""
    st.session_state["f_rut"] = ""
    st.session_state["f_edad"] = None
    st.session_state["f_sexo"] = None
    st.session_state["f_dias_evolucion"] = None
    st.session_state["f_temperatura"] = None
    st.session_state["f_frecuencia_cardiaca"] = None
    st.session_state["f_crepitantes"] = False
    st.session_state["f_murmullo_disminuido"] = False
    st.session_state["f_asma"] = False
    st.session_state["f_sat_o2"] = None
    st.session_state["f_frecuencia_respiratoria"] = None
    st.session_state["f_rinorrea_odinofagia"] = False
    st.session_state["f_antipiretico_6h"] = False
    st.session_state["f_compromiso_conciencia"] = False
    st.session_state["f_matidez"] = False
    st.session_state["f_egofonia"] = False
    st.session_state["f_soplo_tubarico"] = False
    st.session_state["f_epoc"] = False
    st.session_state["f_otra_enf_pulmonar"] = False
    st.session_state["f_insuficiencia_cardiaca"] = False
    st.session_state["f_inmunosupresion"] = False
    st.session_state["f_demencia"] = False
    st.session_state["f_usar_slider_cnn"] = False
    st.session_state["f_cnn_pct"] = 0
    st.session_state["f_modo"] = "Antes del informe (Triage: Ficha + Radiografía)"
    st.session_state["f_fuente_pdf"] = "Sin informe por ahora"
    st.session_state["f_texto_libre"] = ""
    st.session_state["f_nonce"] = st.session_state.get("f_nonce", 0) + 1


def init_form_state() -> None:
    if "f_nonce" not in st.session_state:
        st.session_state["f_nonce"] = 0
    if "f_texto_libre" not in st.session_state:
        st.session_state["f_texto_libre"] = ""
    if "f_usar_slider_cnn" not in st.session_state:
        st.session_state["f_usar_slider_cnn"] = not cnn_status["available"]
    if "f_rut" not in st.session_state:
        st.session_state["f_rut"] = rut_aleatorio()
        apply_preset("paracetamol")
    elif st.session_state.pop("regenerar_rut", False):
        st.session_state["f_rut"] = rut_aleatorio()


def apply_preset(key: str) -> None:
    p = PRESETS[key]
    for k, v in p.items():
        st.session_state[f"f_{k}"] = v
    if not st.session_state.get("f_rut"):
        st.session_state["f_rut"] = rut_aleatorio()
    st.session_state["f_usar_slider_cnn"] = not cnn_status["available"]
    st.session_state["f_fuente_pdf"] = "Informe de ejemplo"
    st.session_state["f_texto_libre"] = ""


init_form_state()


def render_keyword_chips(palabras_clave: list[dict]) -> None:
    if not palabras_clave:
        st.caption("Sin palabras clave radiológicas detectadas.")
        return
    style_map = {
        "Definitiva": "background:#fee2e2;color:#b91c1c;border:1px solid #fecaca;",
        "Probable": "background:#ffedd5;color:#c2410c;border:1px solid #fed7aa;",
        "Posible": "background:#fef3c7;color:#b45309;border:1px solid #fde68a;",
        "Sin neumonía": "background:#dcfce7;color:#15803d;border:1px solid #bbf7d0;",
        "Contexto": "background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;",
    }
    html = "".join(
        f'<span class="kw-chip" style="{style_map.get(item["polaridad"], style_map["Contexto"])}">'
        f'{item["termino"]} · {item["polaridad"]}</span>'
        for item in palabras_clave
    )
    st.markdown(html, unsafe_allow_html=True)


def render_signal_bar(prob_rf: float | None, prob_cnn: float | None, prob_nlp: float | None, prob_int: float) -> None:
    etiquetas, valores, colores = [], [], []
    if prob_rf is not None:
        etiquetas.append("Ficha Clínica (Random Forest)")
        valores.append(round(prob_rf * 100, 1))
        colores.append("#0284c7")
    if prob_cnn is not None:
        etiquetas.append("Radiografía (Red Neuronal CNN)")
        valores.append(round(prob_cnn * 100, 1))
        colores.append("#6366f1")
    if prob_nlp is not None:
        etiquetas.append("Informe PDF (Lectura NLP)")
        valores.append(round(prob_nlp * 100, 1))
        colores.append("#0d9488")
    etiquetas.append("RESULTADO COMBINADO")
    valores.append(round(prob_int * 100, 1))
    colores.append("#dc2626" if prob_int >= 0.70 else "#d97706" if prob_int >= 0.38 else "#16a34a")

    fig = go.Figure(
        go.Bar(
            x=valores,
            y=etiquetas,
            orientation="h",
            marker_color=colores,
            text=[f"{v:.0f}%" for v in valores],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 114], title="Probabilidad estimada (%)"),
        yaxis=dict(autorange="reversed"),
        height=225,
        margin=dict(l=6, r=24, t=8, b=28),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def priority_badge(prio: str) -> str:
    cls = {"ROJO": "badge-red", "AMARILLO": "badge-yellow", "VERDE": "badge-green"}.get(prio, "badge-gray")
    ico = {"ROJO": "🔴", "AMARILLO": "🟡", "VERDE": "🟢"}.get(prio, "⚪")
    return f'<span class="badge {cls}">{ico} PRIORIDAD {prio}</span>'


# ============================================================
# BARRA LATERAL SIMPLE (4 PANTALLAS CLARAS)
# ============================================================
st.sidebar.markdown("## 🫁 NEUMOALERT")
st.sidebar.caption("Apoyo inteligente para urgencia respiratoria")

pagina = st.sidebar.radio(
    "¿Qué quieres hacer?",
    [
        "🏠 Evaluar Paciente",
        "🚨 Cola de Urgencias y Alertas",
        "🩻 Historial y Radiólogo",
        "📊 Modelos IA y Resultados",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Estado del sistema**")
st.sidebar.markdown(
    f"""
- **Ficha Heckerling + Random Forest:** 🟢 Listo
- **Lector PDF (4 categorías):** 🟢 Listo
- **Red Neuronal Rx (CNN):** {"🟢 Conectada" if cnn_status["available"] else "🟡 Modo prueba (lista para enchufar)"}
"""
)
st.sidebar.divider()
st.sidebar.caption(
    "⚠️ Prototipo académico Capstone SIC 2026. No reemplaza la radiografía ni la evaluación de un médico."
)

# ============================================================
# ENCABEZADO SUPERIOR
# ============================================================
st.markdown(
    """
<div class="top-bar">
    <div>
        <p class="top-title">🫁 NEUMOALERT</p>
        <p class="top-sub">Ficha clínica de Heckerling + Random Forest · Lectura NLP de informes · Conector Red Neuronal (CNN)</p>
    </div>
    <div class="top-tag">Prototipo Capstone IA</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PANTALLA 1: EVALUAR PACIENTE (SÚPER INTUITIVA Y GUIADA)
# ============================================================
if pagina == "🏠 Evaluar Paciente":
    st.markdown("#### Prueba rápida con un clic")
    b1, b2, b3, b4, b5 = st.columns([1.05, 1.25, 1.0, 1.15, 0.85])
    b1.button("🔴 Neumonía grave (5/5)", width="stretch", on_click=apply_preset, args=("grave",))
    b2.button("🟡 Tomó paracetamol (oculta fiebre)", width="stretch", on_click=apply_preset, args=("paracetamol",))
    b3.button("🟢 Resfrío común (1/5)", width="stretch", on_click=apply_preset, args=("resfrio",))
    b4.button("⚡ Discrepancia (Rx vs Informe)", width="stretch", on_click=apply_preset, args=("discrepancia",))
    b5.button("🧹 Limpiar todo", width="stretch", on_click=limpiar_formulario_cb, help="Reinicia todos los campos, radiografías y PDF")

    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1.22, 1.0], gap="large")
    nonce = st.session_state.get("f_nonce", 0)

    # ---------------- COLUMNA IZQUIERDA: RADIOGRAFÍA VISIBLE + 2 PESTAÑAS ----------------
    with col_form:
        st.markdown("##### 🩻 Radiografía de tórax *(Red Neuronal CNN)*")
        rx_c1, rx_c2 = st.columns(2)
        with rx_c1:
            rx_file = st.file_uploader("Subir radiografía actual (JPG, PNG o DICOM)", type=["jpg", "jpeg", "png", "dcm"], key=f"rx_actual_up_{nonce}")
        with rx_c2:
            rx_antigua = st.file_uploader("Radiografía antigua (opcional)", type=["jpg", "jpeg", "png", "dcm"], key=f"rx_antigua_up_{nonce}")

        if rx_file is not None or rx_antigua is not None:
            p_c1, p_c2 = st.columns(2)
            with p_c1:
                if rx_file is not None:
                    if rx_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        st.image(rx_file, caption=f"Rx actual: {rx_file.name}", width="stretch")
                    else:
                        st.info(f"Archivo DICOM cargado: {rx_file.name}")
            with p_c2:
                if rx_antigua is not None:
                    if rx_antigua.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        st.image(rx_antigua, caption=f"Rx antigua: {rx_antigua.name}", width="stretch")
                    else:
                        st.info(f"Archivo DICOM antiguo: {rx_antigua.name}")

        usar_slider_cnn = st.checkbox(
            "Ingresar/simular porcentaje de la Red Neuronal (CNN)",
            key="f_usar_slider_cnn",
            help="Mientras se enchufa el archivo .keras/.pt de Sebastián, puedes ingresar el % que entrega su modelo en Colab.",
        )
        cnn_manual_prob = None
        if usar_slider_cnn:
            cnn_pct = st.slider("Probabilidad de la Red Neuronal (CNN) (%)", 0, 100, key="f_cnn_pct")
            cnn_manual_prob = cnn_pct / 100.0

        tab1, tab2 = st.tabs(["1️⃣ Ficha del Médico (Heckerling)", "2️⃣ Informe del Radiólogo (PDF / NLP)"])

        with tab1:
            # Identificación básica
            id1, id2, id3, id4 = st.columns([1.45, 0.75, 0.85, 0.85])
            with id1:
                rut_val = st.text_input("RUT (ficticio)", key="f_rut", placeholder="Ej. 26455629-8")
            with id2:
                st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
                st.button("🎲 Nuevo", width="stretch", on_click=generar_nuevo_rut_cb, help="Generar un RUT chileno ficticio válido")
            with id3:
                edad = st.number_input("Edad", min_value=18, max_value=105, step=1, placeholder="Ej. 45", key="f_edad")
            with id4:
                sexo = st.selectbox("Sexo", ["F", "M"], placeholder="Elegir...", key="f_sexo")

            # Criterios de ingreso (Sección A) compactos
            with st.expander("📋 Criterios de ingreso al protocolo (Sección A — deben cumplirse los 3)", expanded=False):
                ia1, ia2, ia3 = st.columns(3)
                c_adulto = ia1.checkbox("18 años o más", value=(edad is not None and edad >= 18))
                c_sint = ia2.checkbox("Síntoma resp. < 14 días o fiebre", value=True)
                c_rx = ia3.checkbox("Se tomará radiografía (RxT)", value=True)
                if not (c_adulto and c_sint and c_rx):
                    st.warning("Según la ficha médica, si alguna respuesta es No, el paciente no entra al estudio Heckerling.")

            st.markdown("##### C. Los 5 criterios del Puntaje de Heckerling *(1 punto por cada Sí)*")
            h1, h2 = st.columns(2)
            with h1:
                temperatura = st.number_input("1. Temperatura axilar (°C) — suma si > 37,8", min_value=35.0, max_value=42.0, step=0.1, placeholder="Ej. 37.5", key="f_temperatura")
                crepitantes = st.checkbox("3. Crepitaciones al auscultar", key="f_crepitantes", help="Ruidos cortos tipo velcro que no desaparecen al toser")
                asma = st.checkbox("5. Tiene diagnóstico de Asma *(NO tener asma suma 1 pt)*", key="f_asma")
            with h2:
                frecuencia_cardiaca = st.number_input("2. Frecuencia cardíaca (lpm) — suma si > 100", min_value=40, max_value=180, step=1, placeholder="Ej. 90", key="f_frecuencia_cardiaca")
                murmullo_disminuido = st.checkbox("4. Murmullo disminuido localizado", key="f_murmullo_disminuido", help="El aire se escucha entrar menos en una zona")

            # Cálculo inmediato del Puntaje Heckerling para mostrar medidor vivo ahí mismo
            if temperatura is not None and frecuencia_cardiaca is not None:
                pt_heck = (
                    int(temperatura > 37.8)
                    + int(frecuencia_cardiaca > 100)
                    + int(bool(crepitantes))
                    + int(bool(murmullo_disminuido))
                    + int(not bool(asma))
                )
                ref_h = HECKERLING_REF_TABLE[pt_heck]
                st.markdown(
                    f"""
                <div class="heck-banner">
                    <b>Puntaje de Heckerling actual: {pt_heck} / 5</b> · <span>{ref_h['zona']}</span><br>
                    <small style="color:#334155;">{ref_h['detalle']}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                <div class="heck-banner" style="background:#f8fafc;border-color:#cbd5e1;border-left-color:#94a3b8;">
                    <b>Puntaje de Heckerling: pendiente</b><br>
                    <small style="color:#475569;">Ingresa Temperatura y Frecuencia cardíaca para calcular el puntaje (0 a 5).</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("##### D. Datos adicionales clave del médico *(complementan a Heckerling)*")
            d1, d2, d3 = st.columns(3)
            with d1:
                sat_o2 = st.number_input("Saturación O₂ (%)", min_value=60.0, max_value=100.0, step=1.0, placeholder="Ej. 96", key="f_sat_o2")
            with d2:
                fr = st.number_input("Frec. Respiratoria (rpm)", min_value=8, max_value=55, step=1, placeholder="Ej. 20", key="f_frecuencia_respiratoria")
            with d3:
                dias_evol = st.number_input("Días con síntomas", min_value=1, max_value=14, step=1, placeholder="Ej. 3", key="f_dias_evolucion")

            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                rinorrea = st.checkbox("Rinorrea / dolor garganta", key="f_rinorrea_odinofagia", help="Apunta a resfrío viral (en contra de neumonía)")
            with ex2:
                antipiretico = st.checkbox("Tomó antipirético (≤ 6 h)", key="f_antipiretico_6h", help="Paracetamol/ibuprofeno puede esconder la fiebre")
            with ex3:
                conciencia = st.checkbox("Compromiso conciencia", key="f_compromiso_conciencia", help="Glasgow < 15 o desorientación nueva")

            with st.expander("➕ Enfermedades previas (Sección B) y signos poco frecuentes (Matidez / Egofonía / Soplo)", expanded=False):
                st.caption("Marca solo si están presentes en el paciente:")
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.markdown("**Enfermedades previas**")
                    epoc = st.checkbox("EPOC", key="f_epoc")
                    otra_pulm = st.checkbox("Otra enf. pulmonar (fibrosis/TBC/cáncer)", key="f_otra_enf_pulmonar")
                    icc = st.checkbox("Insuficiencia cardíaca (ICC)", key="f_insuficiencia_cardiaca")
                    inmuno = st.checkbox("Inmunosupresión", key="f_inmunosupresion")
                    demencia = st.checkbox("Demencia", key="f_demencia")
                with b_col2:
                    st.markdown("**Signos focales al examen físico**")
                    matidez = st.checkbox("Matidez a la percusión", key="f_matidez")
                    egofonia = st.checkbox("Egofonía", key="f_egofonia")
                    soplo = st.checkbox("Soplo tubárico", key="f_soplo_tubarico")

        with tab2:
            modo_eval = st.radio(
                "¿En qué momento estás evaluando al paciente?",
                [
                    "Con informe del radiólogo (Doble Chequeo)",
                    "Antes del informe (Triage: Ficha + Radiografía)",
                ],
                key="f_modo",
                horizontal=True,
            )

            st.markdown("##### 📄 Informe del radiólogo en PDF *(Lectura NLP en 4 categorías)*")
            fuente_pdf = st.radio(
                "¿Qué informe quieres analizar?",
                ["Informe de ejemplo", "Subir un PDF propio", "Sin informe por ahora"],
                key="f_fuente_pdf",
                horizontal=True,
            )

            pdf_bytes = None
            pdf_name = None
            pdf_up = None

            if fuente_pdf == "Informe de ejemplo":
                opciones_pdf = sorted(p.name for p in SAMPLE_PDF_DIR.glob("*.pdf")) if SAMPLE_PDF_DIR.exists() else []
                if opciones_pdf:
                    if st.session_state.get("f_pdf_file") not in opciones_pdf:
                        st.session_state["f_pdf_file"] = opciones_pdf[0]
                    pdf_name = st.selectbox(
                        "Elige un informe de prueba",
                        opciones_pdf,
                        format_func=lambda fn: SAMPLE_PDF_LABELS.get(fn, fn),
                        key="f_pdf_file",
                    )
                    pdf_bytes = (SAMPLE_PDF_DIR / pdf_name).read_bytes()
            elif fuente_pdf == "Subir un PDF propio":
                pdf_up = st.file_uploader("Selecciona tu archivo PDF", type=["pdf"], key=f"pdf_up_{nonce}")
                if pdf_up is not None:
                    pdf_bytes = pdf_up.getvalue()
                    pdf_name = pdf_up.name

            texto_libre = st.text_area(
                "O pega aquí el texto del informe (opcional)",
                height=72,
                key="f_texto_libre",
                placeholder="Ej.: Consolidación homogénea en base derecha con broncograma aéreo...",
            )

    # ---------------- EJECUCIÓN EN VIVO DE LOS 3 MOTORES ----------------
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
    cnn_res = predict_cnn(image_source=rx_bytes, manual_cnn_prob=cnn_manual_prob)

    if pdf_bytes is not None:
        nlp_res = analizar_pdf_completo(pdf_bytes)
    elif texto_libre.strip():
        nlp_res = analizar_informe_nlp(texto_libre.strip())
    else:
        nlp_res = {
            "es_valido_paciente": False,
            "es_antecedente_previo": False,
            "tipo_documento": "SIN DOCUMENTO PDF",
            "prob_nlp": None,
            "clase_nlp": "Sin informe",
            "hallazgos": [],
            "palabras_clave": [],
            "resumen": "No se adjuntó informe PDF.",
        }

    hay_alguna_senal = (rf_res is not None) or (cnn_res["prob_cnn"] is not None) or bool(nlp_res.get("es_valido_paciente"))

    # ---------------- COLUMNA DERECHA: VEREDICTO CLARO SIEMPRE VISIBLE ----------------
    with col_result:
        if not hay_alguna_senal:
            st.markdown(
                """
            <div class="verdict-box" style="background:#f8fafc;border-color:#cbd5e1;">
                <div class="verdict-kicker">Formulario limpio · Listo para evaluar</div>
                <div class="verdict-main" style="font-size:1.45rem;color:#334155;">Sin datos ingresados todavía</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            fusion = fusionar_predicciones(
                datos_clinicos=datos_paciente,
                prob_rf=rf_res["prob_rf"] if rf_res is not None else None,
                prob_cnn=cnn_res["prob_cnn"],
                prob_nlp=nlp_res["prob_nlp"] if nlp_res.get("es_valido_paciente") else None,
                modo=modo_eval,
                es_antecedente_previo=bool(nlp_res.get("es_antecedente_previo", False)),
            )

            v_cls = {"ROJO": "verdict-red", "AMARILLO": "verdict-yellow", "VERDE": "verdict-green"}[fusion["prioridad"]]
            st.markdown(
                f"""
            <div class="verdict-box {v_cls}">
                <div class="verdict-kicker">Resultado Integrado NEUMOALERT · {fusion['modo_aplicado']}</div>
                <div class="verdict-main">{fusion['prob_integrada']:.0%} &nbsp; {priority_badge(fusion['prioridad'])}</div>
                <div class="verdict-text"><b>Recomendación:</b> {fusion['accion_sugerida']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Alerta clara de concordancia / discrepancia
            conc = fusion["concordancia"]
            if conc["color"] == "rojo":
                st.error(f"**⚠️ {conc['estado']}:** {conc['mensaje']}")
            elif conc["color"] == "amarillo":
                st.warning(f"**🔎 {conc['estado']}:** {conc['mensaje']}")
            elif conc["color"] == "verde":
                st.success(f"**✅ {conc['estado']}:** {conc['mensaje']}")
            else:
                st.info(f"**ℹ️ {conc['estado']}:** {conc['mensaje']}")

            # 3 mini-tarjetas simples
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                rf_num = f"{rf_res['prob_rf']:.0%}" if rf_res is not None else "—"
                rf_sub = f"Heckerling: <b>{rf_res['heckerling']['puntaje']}/5</b>" if rf_res is not None else "Completa la ficha"
                st.markdown(
                    f"""
                <div class="mini-card">
                    <div class="mini-title">📋 Clínica (RF)</div>
                    <div class="mini-num">{rf_num}</div>
                    <div class="mini-sub">{rf_sub}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with mc2:
                cnn_str = f"{cnn_res['prob_cnn']:.0%}" if cnn_res["prob_cnn"] is not None else "—"
                st.markdown(
                    f"""
                <div class="mini-card">
                    <div class="mini-title">🩻 Radiografía</div>
                    <div class="mini-num">{cnn_str}</div>
                    <div class="mini-sub">{"CNN real" if cnn_status["available"] else "Valor CNN (prueba)"}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with mc3:
                nlp_str = f"{nlp_res['prob_nlp']:.0%}" if nlp_res.get("prob_nlp") is not None else "—"
                st.markdown(
                    f"""
                <div class="mini-card">
                    <div class="mini-title">📄 Informe PDF</div>
                    <div class="mini-num" style="font-size:1.18rem; padding-top:0.2rem;">{nlp_res['clase_nlp']}</div>
                    <div class="mini-sub">Score NLP: <b>{nlp_str}</b></div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            render_signal_bar(
                rf_res["prob_rf"] if rf_res is not None else None,
                cnn_res["prob_cnn"],
                nlp_res["prob_nlp"] if (nlp_res.get("es_valido_paciente") and not nlp_res.get("es_antecedente_previo") and "Antes" not in modo_eval) else None,
                fusion["prob_integrada"],
            )

            # Explicación en español cotidiano
            with st.expander("💡 ¿Por qué dio este resultado? (Explicación clínica + palabras del PDF)", expanded=True):
                if rf_res is not None:
                    exp = rf_res["explicacion"]
                    if exp["a_favor"]:
                        st.markdown("**Señales a favor de sospecha:** " + " · ".join(exp["a_favor"][:5]))
                    if exp["en_contra"]:
                        st.markdown("**Señales en contra de neumonía:** " + " · ".join(exp["en_contra"]))
                    for nota in exp["notas"]:
                        st.info(nota)
                else:
                    st.caption("Completa la ficha clínica (edad, sexo y signos vitales) para incluir el análisis del Random Forest.")

                if nlp_res.get("tipo_documento") == "PROTOCOLO O GUÍA CLÍNICA":
                    st.warning("📄 El PDF cargado es una guía médica teórica, no un informe de paciente (filtrado automáticamente).")
                elif nlp_res.get("palabras_clave"):
                    st.markdown("**Palabras clave detectadas en el informe PDF:**")
                    render_keyword_chips(nlp_res["palabras_clave"])

            if nlp_res.get("hallazgos"):
                with st.expander(f"🔎 Ver frases exactas leídas del PDF ({pdf_name or 'texto'})", expanded=False):
                    df_h = pd.DataFrame(
                        [
                            {"Categoría": h["estado"], "Palabras clave": ", ".join(h["palabras_clave"]) or "—", "Frase": h["oracion"]}
                            for h in nlp_res["hallazgos"]
                        ]
                    )
                    st.dataframe(df_h, width="stretch", hide_index=True)

            # Guardar en base de datos
            if st.button("💾 Guardar paciente en la Cola de Urgencias", type="primary", width="stretch"):
                rut_clean = rut_val.strip().upper().replace(".", "")
                if not ficha_completa:
                    st.warning("Completa primero todos los datos de la ficha clínica (edad, sexo y signos vitales) antes de guardar.")
                elif not re.fullmatch(r"\d{7,8}-[\dK]", rut_clean):
                    st.error("Formato de RUT inválido (ej. 16482910-3). Usa el botón '🎲 Nuevo'.")
                else:
                    r_num, dv = rut_clean.split("-")
                    if rut_dv(r_num) != dv:
                        st.error("El dígito verificador del RUT no coincide. Haz clic en '🎲 Nuevo'.")
                    else:
                        try:
                            ruta_rx = guardar_archivo(rx_file, rut_clean, "rx_actual")
                            ruta_rx_ant = guardar_archivo(rx_antigua, rut_clean, "rx_antigua")
                            ruta_pdf = guardar_archivo(pdf_up, rut_clean, "informe_pdf") if pdf_up else (
                                str(SAMPLE_PDF_DIR / pdf_name) if (fuente_pdf == "Informe de ejemplo" and pdf_name) else None
                            )
                            fecha = datetime.now()
                            kw_str = ", ".join(f"{k['termino']} ({k['polaridad']})" for k in nlp_res.get("palabras_clave", []))
                            txt_doc = texto_libre.strip() or (
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
                                            prediccion_ia="Referencia histórica",
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
                            st.success(f"✅ Paciente **{rut_clean}** guardado en urgencias (Prioridad {fusion['prioridad']}).")
                            st.session_state["regenerar_rut"] = True
                        except SQLAlchemyError as exc:
                            st.error(f"No se pudo guardar (puede que el RUT ya exista). Haz clic en '🎲 Nuevo'. Detalle: {exc}")


# ============================================================
# PANTALLA 2: COLA DE URGENCIAS Y ALERTAS
# ============================================================
elif pagina == "🚨 Cola de Urgencias y Alertas":
    st.subheader("🚨 Cola Priorizada de Urgencias y Alertas Activas")
    df_p = obtener_registros(engine, pacientes)
    df_a = obtener_registros(engine, alertas)

    if df_p.empty:
        st.info("No hay pacientes registrados.")
    else:
        orden = {"ROJO": 0, "AMARILLO": 1, "VERDE": 2}
        df_p["orden"] = df_p["prioridad"].map(orden).fillna(3)
        df_p = df_p.sort_values(["orden", "prob_integrada", "fecha_ingreso"], ascending=[True, False, True])

        rojos = int((df_p["prioridad"] == "ROJO").sum())
        amarillos = int((df_p["prioridad"] == "AMARILLO").sum())
        verdes = int((df_p["prioridad"] == "VERDE").sum())
        discs = int(df_p["concordancia_ia"].fillna("").str.contains("DISCREPANCIA", case=False).sum())

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Pacientes en cola", len(df_p))
        k2.metric("🔴 Urgente (Rojo)", rojos)
        k3.metric("🟡 Preferente (Amarillo)", amarillos)
        k4.metric("🟢 Bajo riesgo (Verde)", verdes)
        k5.metric("⚠️ Discrepancias IA", discs)

        t_cola, t_alertas = st.tabs(["📋 Cola de Pacientes", "🔔 Bandeja de Alertas"])

        with t_cola:
            f1, f2, f3 = st.columns([1.3, 1.1, 1.0])
            filtro_p = f1.multiselect("Mostrar prioridades", ["ROJO", "AMARILLO", "VERDE"], default=["ROJO", "AMARILLO", "VERDE"])
            solo_disc = f2.checkbox("Ver solo casos con discrepancia IA")
            buscar_r = f3.text_input("Buscar por RUT")

            vista = df_p[df_p["prioridad"].isin(filtro_p)].copy()
            if solo_disc:
                vista = vista[vista["concordancia_ia"].fillna("").str.contains("DISCREPANCIA", case=False)]
            if buscar_r:
                vista = vista[vista["id"].str.contains(buscar_r.strip(), case=False, na=False)]

            cols = ["prioridad", "id", "edad", "puntaje_heckerling", "sat_o2", "prob_rf", "prob_cnn", "categoria_rx_nlp", "prob_integrada", "concordancia_ia", "estado"]
            vf = vista[cols].copy()
            for cp in ["prob_rf", "prob_cnn", "prob_integrada"]:
                vf[cp] = vf[cp].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
            vf["puntaje_heckerling"] = vf["puntaje_heckerling"].apply(lambda x: f"{int(x)}/5" if pd.notna(x) else "—")
            vf.columns = ["Prioridad", "RUT", "Edad", "Heckerling", "SatO₂", "Clínica (RF)", "Rx (CNN)", "Informe (NLP)", "Combinado", "Concordancia", "Estado"]
            st.dataframe(vf, width="stretch", hide_index=True)

            g1, g2 = st.columns([1.0, 1.3], gap="large")
            with g1:
                fig_b = px.bar(
                    pd.DataFrame({"Prioridad": ["ROJO", "AMARILLO", "VERDE"], "Pacientes": [rojos, amarillos, verdes]}),
                    x="Prioridad", y="Pacientes", color="Prioridad", text="Pacientes",
                    color_discrete_map={"ROJO": "#dc2626", "AMARILLO": "#d97706", "VERDE": "#16a34a"},
                    title="Pacientes por nivel de prioridad",
                )
                fig_b.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_b, width="stretch")

            with g2:
                df_sc = df_p.dropna(subset=["prob_rf", "prob_cnn"]).copy()
                if not df_sc.empty:
                    df_sc["Clínica RF (%)"] = (df_sc["prob_rf"] * 100).round(1)
                    df_sc["Radiografía CNN (%)"] = (df_sc["prob_cnn"] * 100).round(1)
                    fig_s = px.scatter(
                        df_sc, x="Clínica RF (%)", y="Radiografía CNN (%)", color="prioridad",
                        hover_name="id", hover_data=["puntaje_heckerling", "sat_o2", "concordancia_ia"],
                        color_discrete_map={"ROJO": "#dc2626", "AMARILLO": "#d97706", "VERDE": "#16a34a"},
                        title="Mapa Clínica (Heckerling+RF) vs Radiografía (CNN)",
                    )
                    fig_s.update_layout(xaxis=dict(range=[0, 105]), yaxis=dict(range=[0, 105]), height=280, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_s, width="stretch")

        with t_alertas:
            if df_a.empty:
                st.info("No hay alertas registradas.")
            else:
                ta = df_a.merge(df_p[["id", "edad", "puntaje_heckerling", "sat_o2", "concordancia_ia"]], left_on="paciente_id", right_on="id", how="left").sort_values("fecha", ascending=False)
                st.dataframe(
                    ta[["id_x", "prioridad", "paciente_id", "puntaje_heckerling", "sat_o2", "motivo", "estado_alerta", "fecha"]].rename(
                        columns={"id_x": "ID", "paciente_id": "RUT", "puntaje_heckerling": "Heckerling", "sat_o2": "SatO₂"}
                    ),
                    width="stretch",
                    hide_index=True,
                )
                pend = ta[ta["estado_alerta"].isin(["Pendiente", "Enviada"])]
                if not pend.empty:
                    u1, u2, u3 = st.columns([1.0, 1.2, 1.0])
                    aid = u1.selectbox("Elegir alerta (ID)", pend["id_x"].tolist())
                    nest = u2.selectbox("Cambiar estado a", ["Revisada", "Cerrada", "Enviada", "Pendiente"])
                    u3.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
                    if u3.button("✅ Actualizar alerta", type="primary", width="stretch"):
                        with engine.begin() as conn:
                            conn.execute(alertas.update().where(alertas.c.id == int(aid)).values(estado_alerta=nest))
                        st.rerun()


# ============================================================
# PANTALLA 3: HISTORIAL Y RADIÓLOGO
# ============================================================
elif pagina == "🩻 Historial y Radiólogo":
    st.subheader("🩻 Revisión del Radiólogo, Historial y Descarga de Datos")
    df_p = obtener_registros(engine, pacientes)

    if df_p.empty:
        st.warning("Todavía no hay pacientes registrados.")
    else:
        rut_sel = st.selectbox("Selecciona un paciente por RUT", df_p["id"].tolist())
        p = df_p[df_p["id"] == rut_sel].iloc[0]

        st.markdown(f"#### Paciente `{rut_sel}` &nbsp; {priority_badge(str(p['prioridad']))}", unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Heckerling", f"{int(p['puntaje_heckerling'])} / 5")
        m2.metric("SatO₂", f"{p['sat_o2']:.0f}%")
        m3.metric("Temp.", f"{p['temperatura']:.1f} °C")
        m4.metric("FC / FR", f"{int(p['frecuencia_cardiaca'])} / {int(p['frecuencia_respiratoria'])}")
        m5.metric("Combinado", f"{p['prob_integrada']:.0%}" if pd.notna(p["prob_integrada"]) else "—")
        m6.metric("Informe (NLP)", str(p["categoria_rx_nlp"] or "Sin informe"))

        with engine.connect() as conn:
            imgs = pd.read_sql(select(imagenes).where(imagenes.c.paciente_id == rut_sel).order_by(imagenes.c.fecha.desc()), conn)
            diags = pd.read_sql(select(diagnosticos).where(diagnosticos.c.paciente_id == rut_sel).order_by(diagnosticos.c.fecha.desc()), conn)

        c_izq, c_der = st.columns([1.0, 1.15], gap="large")
        with c_izq:
            st.markdown("**Radiografías del paciente**")
            for _, img in imgs.iterrows():
                path = img["ruta_imagen"]
                if path and Path(path).exists() and Path(path).suffix.lower() in (".jpg", ".jpeg", ".png"):
                    st.image(path, width="stretch")
                else:
                    st.info(f"{img['tipo']} · Caso demostrativo sin imagen local adjunta.")
                prob_i = f"{img['prob_ia']:.0%}" if pd.notna(img["prob_ia"]) else "Pendiente"
                st.caption(f"Resultado CNN: {img['prediccion_ia']} ({prob_i})")

        with c_der:
            st.markdown("**Comparación de señales del paciente**")
            render_signal_bar(
                float(p["prob_rf"]) if pd.notna(p["prob_rf"]) else None,
                float(p["prob_cnn"]) if pd.notna(p["prob_cnn"]) else None,
                float(p["prob_nlp"]) if pd.notna(p["prob_nlp"]) else None,
                float(p["prob_integrada"]) if pd.notna(p["prob_integrada"]) else 0.0,
            )

        if not diags.empty:
            st.markdown("**Historial de informes y análisis NLP**")
            st.dataframe(
                diags[["fecha", "tipo_documento", "categoria_rx", "prob_nlp", "palabras_clave", "diagnostico_final", "concordancia"]],
                width="stretch",
                hide_index=True,
            )

        st.divider()
        st.markdown("#### Registrar lectura del radiólogo *(Sección E del estudio)*")
        pdf_rad = st.file_uploader("Subir informe radiológico PDF (opcional, se lee automáticamente)", type=["pdf"], key="rad_pdf")
        txt_rad = st.text_area("O escribir/pegar texto del informe radiológico", height=80)

        nlp_prev = None
        if pdf_rad is not None:
            nlp_prev = analizar_pdf_completo(pdf_rad.getvalue())
        elif txt_rad.strip():
            nlp_prev = analizar_informe_nlp(txt_rad.strip())

        if nlp_prev is not None:
            p_txt = f"{nlp_prev['prob_nlp']:.0%}" if nlp_prev.get("prob_nlp") is not None else "N/A"
            st.info(f"**Lectura NLP automática:** Categoría `{nlp_prev['clase_nlp']}` ({p_txt}) — {nlp_prev['resumen']}")
            render_keyword_chips(nlp_prev.get("palabras_clave", []))

        with st.form("form_rad_v3"):
            r1, r2 = st.columns(2)
            cat_med = r1.selectbox("Categoría final del radiólogo (Sección E)", ["Sin neumonía", "Posible", "Probable", "Definitiva"])
            est_med = r2.selectbox("Estado del paciente", ["Evaluado", "Pendiente de informe", "En evaluación"])
            diag_med = st.text_input("Comentario / diagnóstico final del profesional")
            if st.form_submit_button("💾 Guardar lectura y recalcular prioridad", type="primary"):
                ruta_p = guardar_archivo(pdf_rad, rut_sel, "informe_radiologo") if pdf_rad else None
                prob_n = nlp_prev.get("prob_nlp") if (nlp_prev and nlp_prev.get("es_valido_paciente")) else {"Sin neumonía": 0.06, "Posible": 0.45, "Probable": 0.78, "Definitiva": 0.95}[cat_med]
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
                            texto_medico=txt_rad or (nlp_prev.get("resumen", "") if nlp_prev else f"Clasificado manualmente como {cat_med}."),
                            categoria_rx=cat_med,
                            prob_nlp=prob_n,
                            palabras_clave=kw_n,
                            diagnostico_final=diag_med or f"Radiólogo: {cat_med}",
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
                st.success("Lectura guardada.")
                st.rerun()

        st.divider()
        st.markdown("#### Exportar base de datos completa (para análisis en Excel / Pandas)")
        csv_data = df_p.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Descargar todos los registros en CSV (ficha Heckerling + probabilidades IA)",
            csv_data,
            "neumoalert_heckerling_base.csv",
            "text/csv",
        )


# ============================================================
# PANTALLA 4: MODELOS IA Y RESULTADOS (LISTA PARA EL JURADO)
# ============================================================
elif pagina == "📊 Modelos IA y Resultados":
    st.subheader("📊 Rendimiento de los Modelos IA (Comparación Heckerling vs Random Forest + NLP)")

    if not metrics_data:
        st.warning("No se encontró `models/model_metrics.json`.")
    else:
        rfm = metrics_data["random_forest_clinico"]
        nlpm = metrics_data["nlp_informes_rx"]

        t1, t2, t3 = st.tabs(
            [
                "🌲 1. Heckerling vs Random Forest",
                "📄 2. Lectura NLP (4 categorías)",
                "⚡ 3. Simulador Multimodal y Conector CNN",
            ]
        )

        with t1:
            st.markdown("##### Pregunta clave del estudio: ¿Mejora agregar SatO₂ y las variables del médico al Puntaje de Heckerling?")
            comp_df = pd.DataFrame(rfm["comparacion_modelos"])
            comp_plot = comp_df.melt(id_vars="nombre", value_vars=["roc_auc", "accuracy", "recall", "f1"], var_name="Métrica", value_name="Valor")
            comp_plot["Métrica"] = comp_plot["Métrica"].map({"roc_auc": "AUC-ROC", "accuracy": "Accuracy", "recall": "Recall (Sensibilidad)", "f1": "F1-Score"})
            comp_plot["Porcentaje"] = (comp_plot["Valor"] * 100).round(1)

            fig_comp = px.bar(
                comp_plot, x="Métrica", y="Porcentaje", color="nombre", barmode="group",
                text="Porcentaje",
                color_discrete_sequence=["#94a3b8", "#38bdf8", "#0f4c75"],
                title="Comparación en el mismo conjunto de prueba no visto (Test Set)",
            )
            fig_comp.update_layout(yaxis=dict(range=[0, 108], title="%"), legend_title_text="Modelo", height=340)
            st.plotly_chart(fig_comp, width="stretch")

            # Métricas explicadas en español simple
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC-ROC (Random Forest)", f"{rfm['roc_auc']:.3f}", help="Qué tan bien separa sanos de enfermos (0,5 = azar, 1,0 = perfecto).")
            c2.metric("Accuracy (Aciertos)", f"{rfm['accuracy']:.1%}", help="Porcentaje total de pacientes clasificados correctamente.")
            c3.metric("Recall (Sensibilidad)", f"{rfm['recall']:.1%}", help="De los pacientes que SÍ tenían Rx alterada, qué % logró detectar el modelo.")
            c4.metric("F1-Score", f"{rfm['f1']:.1%}", help="Equilibrio entre detectar casos reales (Recall) y evitar falsas alarmas (Precision).")

            g_fi, g_cm = st.columns([1.25, 1.0], gap="large")
            with g_fi:
                df_fi = pd.DataFrame(rfm["feature_importances"]).head(10)
                fig_fi = px.bar(
                    df_fi, x="importance", y="label", orientation="h",
                    title="Variables más influyentes del Random Forest",
                    color="importance", color_continuous_scale="Teal",
                )
                fig_fi.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=360)
                st.plotly_chart(fig_fi, width="stretch")

            with g_cm:
                fig_cm = px.imshow(
                    rfm["confusion_matrix"], text_auto=True,
                    x=rfm["labels"], y=rfm["labels"],
                    color_continuous_scale="Blues",
                    title="Matriz de Confusión (Random Forest)",
                    labels=dict(x="Predicción del modelo", y="Realidad en RxT"),
                )
                fig_cm.update_layout(height=360)
                st.plotly_chart(fig_cm, width="stretch")

        with t2:
            st.markdown("##### Clasificación automática de informes PDF en las 4 categorías del radiólogo (Sección E)")
            n1, n2, n3 = st.columns(3)
            n1.metric("Accuracy (4 categorías)", f"{nlpm['accuracy']:.1%}")
            n2.metric("F1 Macro", f"{nlpm['f1_macro']:.1%}")
            n3.metric("Validación Cruzada 5-Fold", f"{nlpm['cv_f1_macro_mean']:.1%}")

            c_m, c_t = st.columns([1.05, 1.15], gap="large")
            with c_m:
                fig_nm = px.imshow(
                    nlpm["confusion_matrix"], text_auto=True,
                    x=nlpm["classes"], y=nlpm["classes"],
                    color_continuous_scale="Teal",
                    title="Matriz de Confusión 4×4 (0 errores graves Sin neumonía ↔ Definitiva)",
                    labels=dict(x="Predicción NLP", y="Etiqueta real"),
                )
                fig_nm.update_layout(height=360)
                st.plotly_chart(fig_nm, width="stretch")

            with c_t:
                cls_sel = st.selectbox("Ver frases que más identifican la categoría:", nlpm["classes"])
                df_terms = pd.DataFrame(nlpm["top_terms"][cls_sel])
                fig_terms = px.bar(
                    df_terms, x="peso", y="termino", orientation="h",
                    title=f"Frases clave para '{cls_sel}'",
                    color="peso", color_continuous_scale="Blues",
                )
                fig_terms.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=320)
                st.plotly_chart(fig_terms, width="stretch")

        with t3:
            st.markdown("##### Simulador interactivo para explicar la fusión al jurado")
            s1, s2, s3 = st.columns(3)
            sim_cnn = s1.slider("Radiografía (CNN %)", 0, 100, 88) / 100.0
            sim_rf = s2.slider("Ficha Heckerling + RF (%)", 0, 100, 76) / 100.0
            sim_nlp = s3.slider("Informe PDF (NLP %)", 0, 100, 8) / 100.0

            sim_res = fusionar_predicciones(
                datos_clinicos={"sat_o2": 92, "frecuencia_respiratoria": 25, "temperatura": 38.2, "edad": 65},
                prob_rf=sim_rf, prob_cnn=sim_cnn, prob_nlp=sim_nlp, modo="Con informe",
            )
            render_signal_bar(sim_rf, sim_cnn, sim_nlp, sim_res["prob_integrada"])
            st.markdown(f"{priority_badge(sim_res['prioridad'])} · **{sim_res['concordancia']['estado']}**: {sim_res['concordancia']['mensaje']}", unsafe_allow_html=True)

            st.divider()
            st.markdown("##### 🔌 Conector listo para la Red Neuronal (CNN) de Sebastián")
            if cnn_status["available"]:
                st.success(f"🟢 Modelo CNN conectado: `{cnn_status['path'].name}`")
            else:
                st.info("Cuando Sebastián exporte su CNN en Colab, solo debe guardarla en `models/cnn_pneumonia.keras` (o `.pt`) y se activa automáticamente.")

        st.divider()
        st.caption(f"ℹ️ **Transparencia académica:** {metrics_data.get('nota_academica', '')}")

st.divider()
st.caption("NEUMOALERT v3 · Prototipo académico basado en Heckerling (1990) y Moore (2017) · No es un dispositivo médico.")
