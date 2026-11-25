# 🫁 NEUMOALERT — Plataforma IA Multimodal de Apoyo al Triage y Diagnóstico de Neumonía

Proyecto Capstone · Samsung Innovation Campus (SIC 2026 IA)  
**Equipo:** Sebastián Escobar, Jeancarlo Cuesta, Mary Carmen González, Tabita Molina.

---

## 🎯 Propósito del proyecto

En servicios de urgencia, el personal médico enfrenta sobrecarga y demoras para confirmar radiografías de tórax. **NEUMOALERT** integra una ficha clínica validada médicamente con tres módulos de Inteligencia Artificial:

1. **Ficha Clínica de Heckerling + Random Forest (`src/clinical_rf.py`)**  
   Implementa la ficha clínica propuesta por médico asesor basada en el **Puntaje de Heckerling (1990)** (temperatura > 37,8 °C, FC > 100, crepitaciones, murmullo disminuido, ausencia de asma) más variables modernas (**SatO₂**, **FR**, antipirético en últimas 6 h, rinorrea/odinofagia, matidez, egofonía, soplo tubárico, compromiso de conciencia y comorbilidades).
   - **Comparación en Test Set (no visto):**
     - Puntaje de Heckerling solo: `AUC 0,852` · `Accuracy 81,8%` · `Recall 64,8%`
     - **Random Forest (Heckerling + datos adicionales):** `AUC 0,901` · `Accuracy 86,0%` · `Recall 77,2%` *(+12,4 pts de sensibilidad; `SatO₂` es la variable más influyente con 25,1%)*.
2. **NLP Clínico sobre Informes Radiológicos PDF (`src/nlp_radiology.py`)**  
   Extrae texto con PyMuPDF (+ respaldo OCR opcional), bloquea automáticamente guías/protocolos teóricos, detecta palabras clave radiológicas respetando negaciones y clasifica informes en las **4 categorías exactas de la Sección E del estudio**: `Sin neumonía`, `Posible`, `Probable` y `Definitiva` (**95,5% Accuracy y F1 Macro**, 0 errores graves entre `Sin neumonía` y `Definitiva`).
3. **Conector Red Neuronal Convolucional / CNN (`src/cnn_adapter.py`)**  
   Adaptador *plug-and-play* para conectar la CNN de radiografías de tórax (`models/cnn_pneumonia.keras`, `.h5` o `.pt`), más modo de prueba interactivo.
4. **Motor de Fusión Multimodal (`src/multimodal_fusion.py`)**  
   - **Antes del informe (Triage):** une Ficha Clínica (Random Forest) + Radiografía (CNN).
   - **Con informe del radiólogo (Doble Chequeo):** suma el NLP del PDF y levanta alertas de **`DISCREPANCIA CRÍTICA`** si la radiografía y el texto del informe se contradicen.

---

## 📁 Estructura del repositorio

```text
neumoalert/
├── app.py                              # App web Streamlit (4 pantallas intuitivas + 4 casos demo 1 clic)
├── requirements.txt                    # Dependencias Python
├── data/
│   ├── clinical_pneumonia_dataset.csv  # Dataset clínico Heckerling + Sección D (2.500 registros sintéticos)
│   ├── radiology_reports_dataset.csv   # Informes radiológicos en 4 categorías (1.000 registros sintéticos)
│   └── sample_pdfs/                    # 5 PDFs ficticios listos para prueba en vivo
├── models/
│   ├── rf_clinical_model.joblib        # Pipeline Random Forest entrenado
│   ├── nlp_radiology_model.joblib      # Pipeline NLP TF-IDF + Regresión Logística (4 categorías)
│   ├── model_metrics.json              # Métricas comparativas en Test Set
│   └── cnn_adapter_README.md           # Guía para conectar el modelo CNN (.keras / .pt)
├── src/
│   ├── clinical_rf.py                  # Puntaje Heckerling + inferencia y explicabilidad Random Forest
│   ├── nlp_radiology.py                # Lectura PDF + reglas clínicas + modelo NLP 4 categorías
│   ├── cnn_adapter.py                  # Conector plug-and-play para CNN
│   ├── multimodal_fusion.py            # Fusión multimodal, concordancia y semáforo
│   └── database.py                     # Persistencia SQLAlchemy (SQLite / PostgreSQL)
├── scripts/
│   ├── generate_datasets.py            # Generador reproducible de datasets y PDFs de prueba
│   └── train_models.py                 # Entrenamiento y comparación Heckerling vs RF + NLP
└── notebooks/
    ├── 01_Modelo_Random_Forest_Checklist.ipynb
    └── 02_Modelo_NLP_Informes_Rx_y_Fusion.ipynb
```

---

## 🚀 Cómo ejecutar localmente (Windows PowerShell)

```powershell
# 1. Crear entorno virtual e instalar dependencias
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt

# 2. (Opcional) Regenerar datasets y reentrenar modelos
.\.venv\Scripts\python.exe scripts\generate_datasets.py
.\.venv\Scripts\python.exe scripts\train_models.py

# 3. Iniciar la aplicación web
.\.venv\Scripts\streamlit.exe run app.py
```

Para conectar **PostgreSQL**:
```powershell
$env:DATABASE_URL="postgresql+psycopg2://usuario:clave@localhost:5432/neumoalert"
.\.venv\Scripts\streamlit.exe run app.py
```

---

## ⚠️ Nota de transparencia académica y ética

- Los datasets tabulares y de informes incluidos en `data/` son **sintéticos**, calibrados con el Puntaje de Heckerling (*Ann Intern Med*, 1990) y predictores de Moore et al. (*Eur Respir J*, 2017), para respetar la privacidad médica mientras se recolectan fichas reales. El modelo de imágenes médicas (CNN) utiliza el dataset público *Chest X-Ray Pneumonia*.
- **NEUMOALERT es un prototipo académico.** No es un dispositivo médico ni sustituye el criterio de profesionales de la salud.
