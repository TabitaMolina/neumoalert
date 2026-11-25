# 🔌 Cómo conectar la Red Neuronal (CNN) de Sebastián en NEUMOALERT

El archivo `src/cnn_adapter.py` ya está preparado para detectar y usar automáticamente el modelo CNN entrenado en radiografías de tórax.

## Paso 1: Exportar el modelo desde Google Colab

### Si el modelo fue entrenado con TensorFlow / Keras:
```python
model.save("cnn_pneumonia.keras")  # también sirve .h5
```

### Si el modelo fue entrenado con PyTorch:
```python
import torch
torch.save(model, "cnn_pneumonia.pt")  # también sirve .pth
```

## Paso 2: Copiar el archivo en esta carpeta (`models/`)
Deja el archivo aquí:
- `models/cnn_pneumonia.keras` (o `models/cnn_pneumonia.h5`)
- o `models/cnn_pneumonia.pt` (o `models/cnn_pneumonia.pth`)

## Paso 3: Instalar la librería correspondiente (solo si se corre localmente)
- Para Keras: `pip install tensorflow`
- Para PyTorch: `pip install torch torchvision`

Al reiniciar `streamlit run app.py`, la barra lateral cambiará automáticamente a **🟢 Modelo real (KERAS / PYTORCH)** y cada radiografía subida en la pestaña A pasará por la CNN real y se fusionará con el Random Forest y el NLP clínico.
