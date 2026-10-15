import base64
import json
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from landmark_recognition.inference import load_checkpoint_model, prepare_checkpoint_features
from landmark_recognition.class_labels import display_class_name
from landmark_recognition.mediapipe_support import holistic_result_to_array


def run_colab_webcam(checkpoint_path: str | Path) -> None:
    try:
        import mediapipe as mp
        from google.colab.output import eval_js
        from IPython.display import HTML, Javascript, display
    except ImportError as error:
        raise ImportError(
            "Esta función requiere Google Colab y MediaPipe. Ejecuta la celda de instalación "
            "actualizada del notebook y reinicia el entorno de ejecución de Colab."
        ) from error

    if not hasattr(mp, "solutions"):
        mediapipe_version = getattr(mp, "__version__", "desconocida")
        raise RuntimeError(
            f"MediaPipe {mediapipe_version} no incluye la API `mp.solutions` que usa este "
            "extractor. Ejecuta `%pip install -q --force-reinstall --no-deps "
            "mediapipe==0.10.21`, reinicia el entorno de ejecución de Colab y vuelve a "
            "ejecutar las celdas de configuración y webcam."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    frame_count = int(checkpoint["frame_count"])
    class_names = checkpoint["class_names"]

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )
    camera_id = f"landmark_webcam_{uuid.uuid4().hex}"
    display(HTML(
        f'<div id="{camera_id}">'
        '<video autoplay muted playsinline style="max-width:480px;width:100%;"></video>'
        '<p>Solicitando acceso a la cámara…</p>'
        '<button type="button">Detener cámara</button>'
        "</div>"
    ))
    display(Javascript(
        f"""
        (async () => {{
          const root = document.getElementById({camera_id!r});
          const video = root.querySelector("video");
          const status = root.querySelector("p");
          const button = root.querySelector("button");
          const prefix = {camera_id!r};
          window[prefix + "_ready"] = false;
          window[prefix + "_running"] = false;
          try {{
            const stream = await navigator.mediaDevices.getUserMedia({{
              video: {{width: {{ideal: 640}}, height: {{ideal: 480}}}},
              audio: false
            }});
            video.srcObject = stream;
            await video.play();
            const canvas = document.createElement("canvas");
            window[prefix + "_frame"] = () => {{
              if (!window[prefix + "_running"] || !video.videoWidth) return null;
              const scale = Math.min(1, 640 / video.videoWidth);
              canvas.width = Math.round(video.videoWidth * scale);
              canvas.height = Math.round(video.videoHeight * scale);
              canvas.getContext("2d").drawImage(video, 0, 0);
              return canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
            }};
            window[prefix + "_update"] = text => {{ status.textContent = text; }};
            window[prefix + "_stop"] = () => {{
              window[prefix + "_running"] = false;
              stream.getTracks().forEach(track => track.stop());
              video.srcObject = null;
              status.textContent = "Cámara detenida.";
            }};
            button.onclick = window[prefix + "_stop"];
            window[prefix + "_running"] = true;
            window[prefix + "_ready"] = true;
            status.textContent = "Cámara activa. Realiza una seña frente a ella.";
          }} catch (error) {{
            window[prefix + "_error"] = String(error);
            status.textContent = "No se pudo iniciar la cámara: " + String(error);
          }}
        }})();
        """
    ))

    ready_function = f"window.{camera_id}_ready"
    for _ in range(150):
        if eval_js(ready_function):
            break
        camera_error = eval_js(f"window.{camera_id}_error || ''")
        if camera_error:
            raise RuntimeError(
                f"No se pudo iniciar la cámara del navegador: {camera_error}. "
                "Comprueba el permiso de cámara y que Colab esté abierto mediante HTTPS."
            )
        time.sleep(0.2)
    else:
        raise TimeoutError("La cámara no respondió dentro de 30 segundos.")

    frame_function = f"window.{camera_id}_frame()"
    update_function = f"window.{camera_id}_update"
    stop_function = f"window.{camera_id}_stop()"
    history = deque(maxlen=frame_count)
    frame_index = 0
    try:
        while eval_js(f"window.{camera_id}_running"):
            encoded_frame = eval_js(frame_function)
            if not encoded_frame:
                time.sleep(0.03)
                continue

            image_bytes = base64.b64decode(encoded_frame)
            frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("El navegador envió un frame de cámara que OpenCV no pudo leer.")

            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_index += 1
            if not result.pose_landmarks and not result.left_hand_landmarks and not result.right_hand_landmarks:
                history.clear()
                eval_js(f"{update_function}('No se detectan manos o cuerpo; ajusta el encuadre.')")
                continue

            history.append(holistic_result_to_array(result))
            if len(history) < frame_count or frame_index % 5:
                continue

            features = prepare_checkpoint_features(np.asarray(history, dtype=np.float32), checkpoint)
            inputs = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(inputs), dim=1)[0]
            top_probabilities, top_indices = probabilities.topk(min(3, len(class_names)))
            predictions = [
                f"{display_class_name(class_names[int(index)])}: {float(probability):.1%}"
                for probability, index in zip(top_probabilities.cpu(), top_indices.cpu())
            ]
            eval_js(f"{update_function}({json.dumps(' | '.join(predictions))})")
    finally:
        holistic.close()
        if eval_js(f"window.{camera_id}_running"):
            eval_js(stop_function)
