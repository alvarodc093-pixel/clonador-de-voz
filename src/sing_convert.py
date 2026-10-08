"""Conversion de voz cantada con Seed-VC (Plachtaa, GPL-3.0).

Fiel a la guia Pon_tu_voz_a_cantar (pasos 5-7):
- Seed-VC desmonta la voz original en 3 cosas (que se dice / que notas /
  quien canta) y la vuelve a montar cambiando el timbre por el tuyo.
- Si la melodia queda fuera de tu registro, se transpone de 12 en 12
  semitonos (una octava) para seguir encajando con el instrumental.
- Sin GPU: trabajar con fragmentos de 10-15 s y PASOS=10.

Requiere la carpeta `seed-vc` junto a la raiz del proyecto:
    git clone https://github.com/Plachtaa/seed-vc.git
y el entorno de canto (Python 3.10):
    pip install -r requirements-canto.txt

Carga perezosa: importar este modulo NO carga el modelo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import torch
except ImportError:
    torch = None  # entorno sin modelos: solo fallan convertir()/cargar_motor()

FS = 44_100

_motor = None


def seedvc_disponible() -> bool:
    """True si la carpeta seed-vc esta junto al proyecto."""
    from .config import ROOT

    return ((ROOT / "seed-vc" / "seed_vc_wrapper.py").exists()
            or (Path("seed-vc") / "seed_vc_wrapper.py").exists())


def dispositivo() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def pasos_por_defecto() -> int:
    return 10 if dispositivo() == "cpu" else 30


def cargar_motor(device: str | None = None):
    """Carga SeedVCWrapper una sola vez. Error claro si falta la carpeta."""
    global _motor
    if _motor is not None:
        return _motor
    import sys

    from .config import ROOT

    for candidato in (ROOT / "seed-vc", Path("seed-vc")):
        if (candidato / "seed_vc_wrapper.py").exists():
            sys.path.insert(0, str(candidato.resolve()))
            break
    else:
        raise RuntimeError(
            "No se encontro la carpeta seed-vc junto al proyecto. "
            "Clonala con: git clone https://github.com/Plachtaa/seed-vc.git"
        )
    from seed_vc_wrapper import SeedVCWrapper

    _motor = SeedVCWrapper(device=torch.device(device or dispositivo()))
    # Fix Mac con chip Apple: el detector de tono devuelve float64.
    try:
        _detectar = _motor.rmvpe.infer_from_audio
        _motor.rmvpe.infer_from_audio = (
            lambda *a, **k: _detectar(*a, **k).astype("float32")
        )
    except Exception:
        pass
    return _motor


def convertir(fuente: str | Path, referencia: str | Path,
              semitonos: int = 0, pasos: int | None = None,
              guardar: str | Path | None = None) -> np.ndarray:
    """Voz de `fuente` cantando con el timbre de `referencia`.

    - `semitonos`: 0 = tono original, -12 = una octava mas grave,
      12 = una octava mas aguda.
    - Devuelve audio mono float32 a 44.1 kHz.
    """
    motor = cargar_motor()
    pasos = pasos if pasos is not None else pasos_por_defecto()
    conversion = motor.convert_voice(
        str(fuente), str(referencia), diffusion_steps=pasos,
        f0_condition=True, auto_f0_adjust=False,
        pitch_shift=semitonos, stream_output=False)
    try:
        while True:
            next(conversion)
    except StopIteration as final:
        audio = np.asarray(final.value, dtype=np.float32)
    if guardar is not None:
        guardar = Path(guardar)
        guardar.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(guardar), audio, FS)
    return audio


def melodia(audio: np.ndarray, sr: int) -> np.ndarray:
    """Altura de la voz en Hz cada centesima de segundo (NaN = silencio)."""
    motor = cargar_motor()
    try:
        import librosa

        audio_16k = librosa.resample(np.asarray(audio, dtype=np.float32),
                                     orig_sr=sr, target_sr=16_000)
    except ImportError:
        audio_16k = np.asarray(audio, dtype=np.float32)
    tono = np.asarray(motor.rmvpe.infer_from_audio(audio_16k, thred=0.03),
                      dtype=np.float32)
    tono[tono < 1] = np.nan
    return tono


def diferencia_registro(voz_cancion: np.ndarray, mi_voz: np.ndarray,
                        sr_cancion: int = FS, sr_mia: int | None = None) -> float:
    """Diferencia en semitonos entre la cancion y tu voz hablada."""
    sr_mia = sr_mia or sr_cancion
    med_cancion = float(np.nanmedian(melodia(voz_cancion, sr_cancion)))
    med_mia = float(np.nanmedian(melodia(mi_voz, sr_mia)))
    return 12 * np.log2(med_cancion / med_mia)


def transposicion_recomendada(diferencia: float) -> int:
    """Regla de la guia: si la cancion queda >8 semitonos arriba, bajar octava."""
    if diferencia > 8:
        return -12
    if diferencia < -8:
        return 12
    return 0
