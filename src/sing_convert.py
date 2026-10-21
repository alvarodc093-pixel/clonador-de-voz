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

import math
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import remuestrear

try:
    import torch
except ImportError:
    torch = None  # entorno sin modelos: solo fallan convertir()/cargar_motor()

FS = 44_100

_motor = None
_cerrojo = threading.Lock()


_MODULOS_CANTO = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "munch": "munch",
    "dac": "descript-audio-codec",
    "einops": "einops",
    "transformers": "transformers",
    "scipy": "scipy",
    "librosa": "librosa",
    "pydub": "pydub",
    "yaml": "pyyaml",
}


def dependencias_seedvc() -> list[str]:
    """Lista de paquetes del entorno de canto que faltan (por su nombre pip)."""
    import importlib.util

    faltan: list[str] = []
    for modulo, paquete in _MODULOS_CANTO.items():
        try:
            disponible = importlib.util.find_spec(modulo) is not None
        except (ImportError, ValueError):
            disponible = False
        if not disponible:
            faltan.append(paquete)
    return faltan


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


def _parchear_bigvgan_hub() -> None:
    """Compatibiliza el BigVGAN de seed-vc con huggingface_hub >= 1.0.

    En hub 1.x, ``PyTorchModelHubMixin.from_pretrained`` ya no pasa
    ``proxies``/``resume_download`` a ``_from_pretrained`` y
    ``hf_hub_download`` tampoco los acepta; el wrapper de seed-vc los
    declara obligatorios y la carga del vocoder muere con TypeError.
    """
    import inspect

    try:
        from huggingface_hub import hf_hub_download as _fhd
        from modules.bigvgan import bigvgan as bg
    except Exception:
        return

    params = set(inspect.signature(_fhd).parameters)
    if {"proxies", "resume_download"} <= params:
        return  # hub antiguo: compatible sin parche

    def _fhd_compat(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        return _fhd(*args, **kwargs)

    bg.hf_hub_download = _fhd_compat

    original = bg.BigVGAN._from_pretrained.__func__

    def _from_pretrained(cls, *, model_id, revision=None, cache_dir=None,
                         force_download=False, local_files_only=False,
                         token=None, proxies=None, resume_download=False,
                         **kwargs):
        return original(cls, model_id=model_id, revision=revision,
                        cache_dir=cache_dir, force_download=force_download,
                        proxies=proxies, resume_download=resume_download,
                        local_files_only=local_files_only, token=token,
                        **kwargs)

    bg.BigVGAN._from_pretrained = classmethod(_from_pretrained)


def cargar_motor(device: str | None = None):
    """Carga SeedVCWrapper una sola vez. Error claro si falta la carpeta."""
    global _motor
    if _motor is not None:
        return _motor
    if torch is None:
        raise RuntimeError(
            "Falta PyTorch en este entorno. Instala el entorno de canto: "
            "pip install -r requirements-canto.txt (Python 3.10)."
        )
    with _cerrojo:
        if _motor is not None:
            return _motor
        from .config import ROOT

        for candidato in (ROOT / "seed-vc", Path("seed-vc")):
            if (candidato / "seed_vc_wrapper.py").exists():
                ruta = str(candidato.resolve())
                if ruta not in sys.path:
                    sys.path.insert(0, ruta)
                break
        else:
            raise RuntimeError(
                "No se encontro la carpeta seed-vc junto al proyecto. "
                "Clonala con: git clone https://github.com/Plachtaa/seed-vc.git"
            )
        _parchear_bigvgan_hub()
        try:
            from seed_vc_wrapper import SeedVCWrapper
        except Exception as e:
            raise RuntimeError(
                "No se pudo importar seed_vc_wrapper. Revisa que el entorno de "
                f"canto esté instalado (requirements-canto.txt). Detalle: {e}"
            ) from e
        try:
            _motor = SeedVCWrapper(device=torch.device(device or dispositivo()))
        except Exception as e:
            raise RuntimeError(
                "No se pudo cargar el modelo de Seed-VC (¿sin conexión para "
                f"descargar los pesos o sin memoria suficiente?). Detalle: {e}"
            ) from e
        # Fix Mac con chip Apple: el detector de tono devuelve float64.
        try:
            _detectar = _motor.rmvpe.infer_from_audio
            _motor.rmvpe.infer_from_audio = (
                lambda *a, **k: _detectar(*a, **k).astype("float32")
            )
        except Exception:
            pass
        return _motor


def _extraer_audio(conversion) -> np.ndarray:
    """Acepta el generador de Seed-VC (o un array) y devuelve un array 1D."""
    es_array = (isinstance(conversion, (np.ndarray, list, tuple))
                or hasattr(conversion, "detach"))
    if es_array or not hasattr(conversion, "__iter__"):
        audio = conversion
    else:
        resultado = None
        iterador = iter(conversion)
        while True:
            try:
                item = next(iterador)
            except StopIteration as final:
                if final.value is not None:
                    resultado = final.value
                break
            if isinstance(item, tuple) and len(item) == 2 and item[1] is not None:
                resultado = item[1]
        audio = resultado
    if audio is None:
        raise RuntimeError("Seed-VC no devolvió audio.")
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    if isinstance(audio, tuple):  # (sr, audio)
        audio = audio[-1]
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
    audio = audio.reshape(-1)
    if audio.size == 0:
        raise RuntimeError("Seed-VC devolvió audio vacío.")
    return audio


def convertir(fuente: str | Path, referencia: str | Path,
              semitonos: int = 0, pasos: int | None = None,
              guardar: str | Path | None = None) -> np.ndarray:
    """Voz de `fuente` cantando con el timbre de `referencia`.

    - `semitonos`: 0 = tono original, -12 = una octava mas grave,
      12 = una octava mas aguda.
    - Devuelve audio mono float32 a 44.1 kHz.
    """
    fuente = Path(fuente).expanduser()
    referencia = Path(referencia).expanduser()
    if not fuente.exists():
        raise FileNotFoundError(f"No existe la voz de la canción: {fuente}")
    if not referencia.exists():
        raise FileNotFoundError(f"No existe tu voz de referencia: {referencia}")
    semitonos = int(semitonos)
    if not -24 <= semitonos <= 24:
        raise ValueError("semitonos debe estar entre -24 y 24.")
    pasos = int(pasos) if pasos is not None else pasos_por_defecto()
    if pasos < 1:
        raise ValueError("pasos debe ser al menos 1.")
    motor = cargar_motor()
    conversion = motor.convert_voice(
        str(fuente), str(referencia), diffusion_steps=pasos,
        f0_condition=True, auto_f0_adjust=False,
        pitch_shift=semitonos, stream_output=False)
    audio = _extraer_audio(conversion)
    if guardar is not None:
        guardar = Path(guardar)
        guardar.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(guardar), audio, FS)
    return audio


def _mediana_tono(audio: np.ndarray, sr: int) -> float:
    valores = np.asarray(melodia(audio, sr), dtype=np.float32)
    valores = valores[np.isfinite(valores) & (valores > 0)]
    if valores.size == 0:
        raise ValueError("No se detectó tono en el audio (¿silencio o ruido?).")
    return float(np.median(valores))


def melodia(audio: np.ndarray, sr: int) -> np.ndarray:
    """Altura de la voz en Hz cada centesima de segundo (NaN = silencio)."""
    sr = int(sr)
    if sr <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).ravel(),
                          nan=0.0, posinf=0.0, neginf=0.0)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    motor = cargar_motor()
    if not hasattr(motor, "rmvpe"):
        raise RuntimeError("El motor de Seed-VC no expone el detector de tono (rmvpe).")
    audio_16k = remuestrear(audio, sr, 16_000)
    tono = np.asarray(motor.rmvpe.infer_from_audio(audio_16k, thred=0.03),
                      dtype=np.float32).reshape(-1)
    tono[tono < 1] = np.nan
    return tono


def diferencia_registro(voz_cancion: np.ndarray, mi_voz: np.ndarray,
                        sr_cancion: int = FS, sr_mia: int | None = None) -> float:
    """Diferencia en semitonos entre la cancion y tu voz hablada."""
    sr_mia = int(sr_mia or sr_cancion)
    med_cancion = _mediana_tono(voz_cancion, sr_cancion)
    med_mia = _mediana_tono(mi_voz, sr_mia)
    if med_cancion <= 0 or med_mia <= 0:
        raise ValueError("No se pudo medir el registro de una de las voces.")
    return float(12 * math.log2(med_cancion / med_mia))


def transposicion_recomendada(diferencia: float) -> int:
    """Regla de la guia: si la cancion queda >8 semitonos arriba, bajar octava."""
    try:
        diferencia = float(diferencia)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(diferencia):
        return 0
    if diferencia > 8:
        return -12
    if diferencia < -8:
        return 12
    return 0
