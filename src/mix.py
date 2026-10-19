"""Mezcla final: voz (clonada/convertida) + base musical de fondo.

La receta de la guía de canto (paso 8):
1. Igualar el nivel de la voz al de referencia.
2. Recortar a la misma duración.
3. Evitar saturación (pico 0.98) y dejar fade corto anti-clics.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import fundido_cruzado_simple, remuestrear


def _a_estereo(audio: np.ndarray) -> np.ndarray:
    """Normaliza cualquier array de audio a (2, n) float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    elif audio.ndim != 2:
        raise ValueError(f"Forma de audio no soportada: {audio.shape}")
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def cargar_cancion_estereo(ruta: str | Path, sr_objetivo: int = 44_100):
    """Carga mp3/wav en estéreo (2, n) remuestreado a `sr_objetivo`."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de audio: {ruta}")
    sr_objetivo = int(sr_objetivo)
    if sr_objetivo <= 0:
        raise ValueError("sr_objetivo debe ser positivo.")
    try:
        import librosa

        audio, _ = librosa.load(str(ruta), sr=sr_objetivo, mono=False)
        return _a_estereo(audio), sr_objetivo
    except ImportError:
        if ruta.suffix.lower() in {".mp3", ".m4a", ".aac", ".ogg", ".flac"}:
            raise RuntimeError(
                f"No se puede leer {ruta.name}: instala librosa "
                "(viene en requirements-voz.txt / requirements-canto.txt) "
                "o convierte a WAV primero.")
        audio, sr = sf.read(str(ruta), dtype="float32", always_2d=True)
        return _a_estereo(audio.T), int(sr)


def mezclar(voz_mono: np.ndarray, instrumental: np.ndarray, sr: int = 44_100,
            sr_voz: int | None = None, volumen_voz: float = 1.0,
            nivel_referencia: np.ndarray | None = None,
            guardar: str | Path | None = None) -> np.ndarray:
    """Suma voz mono + instrumental estéreo. Devuelve (2, n).

    - `sr`: frecuencia de la base. Si `sr_voz` difiere (p. ej. voz
      Chatterbox a 24 kHz), la voz se remuestrea antes de sumar.
    - `volumen_voz`: 1.0 = nivel original, 1.3 sube, 0.8 baja.
    - `nivel_referencia`: si se pasa la voz original, iguala RMS a ese nivel
      (como en el cuaderno: la mezcla se hizo para ese nivel).
    """
    sr = int(sr)
    if sr <= 0:
        raise ValueError("La frecuencia de mezcla debe ser positiva.")
    if not np.isfinite(volumen_voz):
        raise ValueError("volumen_voz debe ser un número finito.")

    voz = np.nan_to_num(np.asarray(voz_mono, dtype=np.float32).ravel(),
                        nan=0.0, posinf=0.0, neginf=0.0)
    if voz.size == 0:
        raise ValueError("La pista de voz está vacía.")
    if sr_voz is not None and int(sr_voz) != sr:
        voz = remuestrear(voz, sr_voz, sr)

    inst = _a_estereo(instrumental)
    if inst.shape[1] == 0:
        raise ValueError("La base musical está vacía.")
    n = min(len(voz), inst.shape[1])
    if n == 0:
        raise ValueError("No hay muestras comunes para mezclar.")
    if nivel_referencia is not None:
        ref = np.nan_to_num(
            np.asarray(nivel_referencia, dtype=np.float32).ravel()[:n],
            nan=0.0, posinf=0.0, neginf=0.0)
        if ref.size == 0:
            raise ValueError("La referencia de nivel está vacía.")
        rms_voz = float(np.sqrt((voz[:n] ** 2).mean())) + 1e-9
        rms_ref = float(np.sqrt((ref ** 2).mean())) + 1e-9
        voz = voz * (rms_ref / rms_voz)

    voz = fundido_cruzado_simple(voz, sr)
    mezcla = inst[:, :n] + float(volumen_voz) * voz[:n]
    pico = float(np.abs(mezcla).max()) if mezcla.size else 0.0
    if pico > 0.98:
        mezcla = mezcla * (0.98 / pico)
    if guardar is not None:
        guardar = Path(guardar)
        guardar.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(guardar), mezcla.T, sr)
    return mezcla.astype(np.float32)
