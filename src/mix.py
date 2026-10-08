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


def cargar_cancion_estereo(ruta: str | Path, sr_objetivo: int = 44_100):
    """Carga mp3/wav en estéreo (2, n) remuestreado a `sr_objetivo`."""
    try:
        import librosa

        audio, _ = librosa.load(str(ruta), sr=sr_objetivo, mono=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        return audio, sr_objetivo
    except ImportError:
        if Path(ruta).suffix.lower() == ".mp3":
            raise RuntimeError(
                f"No se puede leer {Path(ruta).name}: para mp3 instala librosa "
                "(viene en requirements-voz.txt / requirements-canto.txt) "
                "o convierte a WAV primero.")
        audio, sr = sf.read(str(ruta), dtype="float32", always_2d=True)
        audio = np.asarray(audio.T, dtype=np.float32)  # (canales, n)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        return audio, int(sr)


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
    voz = np.asarray(voz_mono, dtype=np.float32).ravel()
    if sr_voz is not None and sr_voz != sr:
        voz = remuestrear(voz, sr_voz, sr)
    inst = np.asarray(instrumental, dtype=np.float32)
    if inst.ndim == 1:
        inst = np.stack([inst, inst])
    n = min(len(voz), inst.shape[1])
    if nivel_referencia is not None:
        ref = np.asarray(nivel_referencia, dtype=np.float32).ravel()[:n]
        rms_voz = float(np.sqrt((voz[:n] ** 2).mean())) + 1e-9
        rms_ref = float(np.sqrt((ref ** 2).mean())) + 1e-9
        voz = voz * (rms_ref / rms_voz)
    voz = fundido_cruzado_simple(voz, sr)
    mezcla = inst[:, :n] + float(volumen_voz) * voz[:n]
    pico = float(np.abs(mezcla).max())
    if pico > 0.98:
        mezcla = mezcla * (0.98 / pico)
    if guardar is not None:
        guardar = Path(guardar)
        guardar.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(guardar), mezcla.T, sr)
    return mezcla.astype(np.float32)
