"""Utilidades de audio compartidas por el bot y la app.

Sin dependencias pesadas: solo numpy + soundfile. Todo lo que necesite
torch / librosa / modelos va en su módulo propio con carga perezosa.
"""
from pathlib import Path

import numpy as np
import soundfile as sf


def leer_wav(ruta: str | Path) -> tuple[np.ndarray, int]:
    """Lee un wav como mono float32. Devuelve (audio, sr)."""
    audio, sr = sf.read(str(ruta), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # estéreo -> mono promediando
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, int(sr)


def guardar_wav(ruta: str | Path, audio: np.ndarray, sr: int) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(ruta), np.asarray(audio, dtype=np.float32), sr)
    return ruta


def revisar_grabacion(audio: np.ndarray, sr: int) -> dict:
    """Las 3 comprobaciones de la guía Chatterbox: duración, volumen y pico.

    Devuelve dict con valores + booleano ok + consejos.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    duracion = len(audio) / sr if sr else 0.0
    pico = float(np.abs(audio).max()) if len(audio) else 0.0
    volumen = float(np.sqrt((audio ** 2).mean())) if len(audio) else 0.0
    cheques = [
        {"nombre": f"Duración: {duracion:.1f} s", "ok": duracion >= 8,
         "consejo": "Graba al menos 10-20 segundos." if duracion < 8 else ""},
        {"nombre": f"Volumen medio: {volumen:.3f}", "ok": volumen >= 0.01,
         "consejo": "Se oye muy bajo: acércate al micrófono." if volumen < 0.01 else ""},
        {"nombre": f"Pico máximo: {pico:.2f}", "ok": pico < 0.99,
         "consejo": "La grabación satura: aléjate un poco del micro." if pico >= 0.99 else ""},
    ]
    return {"duracion": duracion, "volumen": volumen, "pico": pico,
            "ok": all(c["ok"] for c in cheques), "cheques": cheques}


def remuestrear(audio: np.ndarray, sr_origen: int, sr_destino: int) -> np.ndarray:
    """Cambia la frecuencia de muestreo. Usa librosa si está, si no interpolación lineal.

    Evita mezclar pistas a distinta frecuencia (p. ej. voz Chatterbox a
    24 kHz sobre base a 44.1 kHz), que sonaría con otro tono y tempo.
    """
    if sr_origen == sr_destino:
        return np.asarray(audio, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sr_origen,
                                target_sr=sr_destino).astype(np.float32)
    except ImportError:
        n_nuevo = int(round(len(audio) * sr_destino / sr_origen))
        x_viejo = np.linspace(0, 1, len(audio))
        x_nuevo = np.linspace(0, 1, n_nuevo)
        return np.interp(x_nuevo, x_viejo, audio).astype(np.float32)


def normalizar_pico(audio: np.ndarray, pico_objetivo: float = 0.89) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    pico = float(np.abs(audio).max())
    if pico < 1e-9:
        return audio
    return (audio * (pico_objetivo / pico)).astype(np.float32)


def fundido_cruzado_simple(audio: np.ndarray, sr: int, ms: int = 10) -> np.ndarray:
    """Pequeño fade-in/out para evitar clics al unir frases o mezclar."""
    n = min(len(audio), int(sr * ms / 1000))
    if n <= 0:
        return audio
    rampa = np.linspace(0, 1, n, dtype=np.float32)
    audio = audio.astype(np.float32).copy()
    audio[:n] *= rampa
    audio[-n:] *= rampa[::-1]
    return audio
