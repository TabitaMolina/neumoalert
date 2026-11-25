"""
Conector plug-and-play para la Red Neuronal Convolucional (CNN) de radiografías de tórax.

Cuando Sebastián termine de entrenar la CNN, solo tiene que guardar su archivo en `models/` con
uno de estos nombres:
- `models/cnn_pneumonia.keras` o `models/cnn_pneumonia.h5` (TensorFlow / Keras)
- `models/cnn_pneumonia.pt` o `models/cnn_pneumonia.pth` (PyTorch)

Mientras ese archivo no exista, el adaptador opera en modo de integración/simulación controlada
para permitir probar y demostrar la fusión multimodal (CNN + Random Forest + NLP) desde hoy.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

KERAS_CANDIDATES = [MODELS_DIR / "cnn_pneumonia.keras", MODELS_DIR / "cnn_pneumonia.h5"]
PYTORCH_CANDIDATES = [MODELS_DIR / "cnn_pneumonia.pt", MODELS_DIR / "cnn_pneumonia.pth"]


def detect_cnn_backend() -> dict[str, Any]:
    """Detecta automáticamente si existe un modelo CNN entrenado en `models/`."""
    for path in KERAS_CANDIDATES:
        if path.exists():
            return {"available": True, "backend": "keras", "path": path}
    for path in PYTORCH_CANDIDATES:
        if path.exists():
            return {"available": True, "backend": "pytorch", "path": path}
    return {"available": False, "backend": "demo", "path": None}


def _preprocess_image_array(image_bytes: bytes, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _demo_estimate_from_image(image_bytes: bytes) -> float:
    """
    Genera una estimación determinística de demostración cuando aún no está conectado
    el archivo de pesos de la CNN. Claramente etiquetada como modo demo.
    """
    digest = hashlib.sha256(image_bytes).digest()
    val = int.from_bytes(digest[:2], "big") / 65535.0
    # Mapea a un rango demostrativo realista [0.22, 0.91]
    return round(0.22 + val * 0.69, 4)


def predict_cnn(
    image_source: bytes | str | Path | None = None,
    manual_cnn_prob: float | None = None,
) -> dict[str, Any]:
    """
    Obtiene la probabilidad de neumonía a partir de la radiografía de tórax.

    Parámetros:
    - image_source: bytes o ruta de imagen (JPG/PNG).
    - manual_cnn_prob: permite ingresar manualmente la probabilidad que arrojó la CNN
      de Sebastián en Colab mientras se integra el archivo de pesos.
    """
    # 1. Si el usuario ingresó manualmente la salida real de la CNN de Sebastián
    if manual_cnn_prob is not None:
        prob = round(float(np.clip(manual_cnn_prob, 0.0, 1.0)), 4)
        return {
            "prob_cnn": prob,
            "prediccion_cnn": "Sospecha de neumonía (CNN)" if prob >= 0.50 else "Sin sospecha de neumonía (CNN)",
            "modo": "CNN_INGRESO_MANUAL_COLAB",
            "detalle": "Probabilidad ingresada desde el modelo CNN externo (ej. Colab de Sebastián).",
        }

    if image_source is None:
        return {
            "prob_cnn": None,
            "prediccion_cnn": "Sin radiografía analizada",
            "modo": "SIN_IMAGEN",
            "detalle": "No se adjuntó radiografía de tórax.",
        }

    if isinstance(image_source, (str, Path)):
        path_obj = Path(image_source)
        if not path_obj.exists():
            return {
                "prob_cnn": None,
                "prediccion_cnn": "Archivo de imagen no disponible",
                "modo": "ERROR_ARCHIVO",
                "detalle": f"No se encontró el archivo {path_obj.name}.",
            }
        image_bytes = path_obj.read_bytes()
    else:
        image_bytes = image_source

    backend_info = detect_cnn_backend()

    # 2. Inferencia real Keras / TensorFlow si el archivo existe y TensorFlow está instalado
    if backend_info["backend"] == "keras":
        try:
            import tensorflow as tf  # type: ignore

            model = tf.keras.models.load_model(str(backend_info["path"]))
            input_shape = model.input_shape
            h = int(input_shape[1]) if input_shape and len(input_shape) >= 3 and input_shape[1] else 224
            w = int(input_shape[2]) if input_shape and len(input_shape) >= 3 and input_shape[2] else 224
            arr = _preprocess_image_array(image_bytes, (w, h))
            batch = np.expand_dims(arr, axis=0)
            pred = model.predict(batch, verbose=0)
            prob = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0][1])
            prob = round(float(np.clip(prob, 0.0, 1.0)), 4)
            return {
                "prob_cnn": prob,
                "prediccion_cnn": "Sospecha de neumonía (CNN)" if prob >= 0.50 else "Sin sospecha de neumonía (CNN)",
                "modo": "CNN_REAL_KERAS",
                "detalle": f"Inferencia real con {backend_info['path'].name}.",
            }
        except Exception as exc:
            prob_demo = _demo_estimate_from_image(image_bytes)
            return {
                "prob_cnn": prob_demo,
                "prediccion_cnn": "Sospecha de neumonía (Modo Demo)" if prob_demo >= 0.50 else "Sin sospecha (Modo Demo)",
                "modo": "MODO_DEMO_FALLBACK",
                "detalle": f"Se encontró modelo Keras pero no pudo cargarse ({exc}). Usando simulación demo.",
            }

    # 3. Inferencia real PyTorch si el archivo existe y torch está instalado
    if backend_info["backend"] == "pytorch":
        try:
            import torch  # type: ignore

            model = torch.load(str(backend_info["path"]), map_location="cpu", weights_only=False)
            model.eval()
            arr = _preprocess_image_array(image_bytes, (224, 224))
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                if out.shape[-1] == 1:
                    prob = float(torch.sigmoid(out)[0, 0].item())
                else:
                    prob = float(torch.softmax(out, dim=-1)[0, 1].item())
            prob = round(float(np.clip(prob, 0.0, 1.0)), 4)
            return {
                "prob_cnn": prob,
                "prediccion_cnn": "Sospecha de neumonía (CNN)" if prob >= 0.50 else "Sin sospecha de neumonía (CNN)",
                "modo": "CNN_REAL_PYTORCH",
                "detalle": f"Inferencia real con {backend_info['path'].name}.",
            }
        except Exception as exc:
            prob_demo = _demo_estimate_from_image(image_bytes)
            return {
                "prob_cnn": prob_demo,
                "prediccion_cnn": "Sospecha de neumonía (Modo Demo)" if prob_demo >= 0.50 else "Sin sospecha (Modo Demo)",
                "modo": "MODO_DEMO_FALLBACK",
                "detalle": f"Se encontró modelo PyTorch pero no pudo cargarse ({exc}). Usando simulación demo.",
            }

    # 4. Modo demo limpio mientras Sebastián entrega el archivo final
    prob_demo = _demo_estimate_from_image(image_bytes)
    return {
        "prob_cnn": prob_demo,
        "prediccion_cnn": "Sospecha de neumonía (Simulación CNN)" if prob_demo >= 0.50 else "Sin sospecha (Simulación CNN)",
        "modo": "MODO_DEMO_SIMULADO",
        "detalle": "Simulación temporal de CNN. Para usar el modelo real, guarda `cnn_pneumonia.keras` o `.pt` en `models/` o ingresa la probabilidad manual.",
    }
