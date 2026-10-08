"""Separación voz / instrumental con Demucs (Meta, MIT).

Fiel a la guía de canto (paso 3): `demucs --two-stems vocals -n htdemucs`.
Se ejecuta como submódulo python (`python -m demucs`) para no depender
del binario `demucs` en el PATH.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def demucs_disponible() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separar(cancion_wav: str | Path, salida: str | Path,
            modelo: str = "htdemucs") -> dict[str, Path]:
    """Separa en voz (vocals.wav) e instrumental (no_vocals.wav).

    `cancion_wav`: fragmento en wav (estéreo 44.1 kHz recomendado).
    `salida`: carpeta donde Demucs escribe htdemucs/<nombre>/...
    Devuelve dict con rutas {voz, instrumental}.
    """
    if not demucs_disponible():
        raise RuntimeError(
            "Demucs no está instalado. Activa el entorno de canto e instala: "
            "pip install -r requirements-canto.txt (Python 3.10)."
        )
    cancion_wav = Path(cancion_wav)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    proceso = subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals",
         "-n", modelo, "-o", str(salida), str(cancion_wav)],
        capture_output=True, text=True)
    if proceso.returncode != 0:
        raise RuntimeError(f"Demucs falló:\n{proceso.stderr[-2000:]}")
    pistas = salida / modelo / cancion_wav.stem
    voz = pistas / "vocals.wav"
    inst = pistas / "no_vocals.wav"
    for p in (voz, inst):
        if not p.exists():
            raise FileNotFoundError(f"Demucs no generó {p}")
    return {"voz": voz, "instrumental": inst}


def _a_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # Demucs escribe las pistas en estéreo
        if audio.shape[0] in (1, 2) and audio.shape[1] not in (1, 2):
            return audio.mean(axis=0).astype(np.float32)
        return audio.mean(axis=1).astype(np.float32)
    return audio.ravel().astype(np.float32)


def cargar_pistas(rutas: dict[str, Path], sr: int = 44_100):
    """Carga voz (mono) e instrumental (estéreo). Devuelve (voz, inst, sr)."""
    try:
        import librosa

        voz, _ = librosa.load(str(rutas["voz"]), sr=sr, mono=True)
        inst, _ = librosa.load(str(rutas["instrumental"]), sr=sr, mono=False)
        inst = np.asarray(inst, dtype=np.float32)
        if inst.ndim == 1:
            inst = np.stack([inst, inst])
        return np.asarray(voz, dtype=np.float32), inst, sr
    except ImportError:
        voz, sr_voz = sf.read(str(rutas["voz"]), dtype="float32", always_2d=False)
        inst, sr_inst = sf.read(str(rutas["instrumental"]), dtype="float32", always_2d=True)
        assert sr_voz == sr_inst, "Las pistas separadas no coinciden en frecuencia"
        return (_a_mono(voz), np.asarray(inst.T, dtype=np.float32), int(sr_voz))
