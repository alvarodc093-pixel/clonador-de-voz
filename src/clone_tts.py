"""Clonación de voz hablada con Chatterbox Multilingual (Resemble AI, MIT).

Fiel a la guía `Archivos/Clona_tu_voz_con_Chatterbox.ipynb`:
- Referencia: 10-20 s de tu voz (mi_voz.wav).
- Estilos: plano / natural / expresivo / teatral
  (parejas exaggeration + cfg_weight).
- Textos largos frase a frase con pausa configurable.
- Semilla fija para repetir resultados + reintento anti-repeticiones.

El modelo pesa ~3 GB y se descarga la primera vez. Este módulo NO lo
carga al importar: usa `cargar_modelo()` (cacheado en Streamlit con
st.cache_resource) para no pagar el coste si solo se extrae audio.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import torch
except ImportError:
    torch = None  # entorno sin modelos: solo fallan hablar()/cargar_modelo()

ESTILOS = {
    "plano": {"exaggeration": 0.25, "cfg_weight": 0.5},
    "natural": {"exaggeration": 0.5, "cfg_weight": 0.5},
    "expresivo": {"exaggeration": 0.7, "cfg_weight": 0.3},
    "teatral": {"exaggeration": 0.9, "cfg_weight": 0.3},
}

IDIOMAS_SOPORTADOS = ["es", "en", "fr", "it", "de", "pt"]


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


def modelo_disponible() -> bool:
    try:
        import chatterbox  # noqa: F401
        return True
    except Exception:
        return False


_modelo = None


def cargar_modelo(device: str | None = None):
    """Carga Chatterbox Multilingual una sola vez (lazy). Lanza error claro si falta."""
    global _modelo
    if _modelo is not None:
        return _modelo
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError as e:
        raise RuntimeError(
            "Chatterbox no está instalado. Activa el entorno de voz e instala: "
            "pip install -r requirements-voz.txt (Python 3.11). "
            f"Detalle: {e}"
        ) from e
    device = device or dispositivo()
    _modelo = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return _modelo


def hablar(texto: str, voz_referencia: str | Path, estilo: str = "natural",
           idioma: str = "es", semilla: int = 7, guardar: str | Path | None = None,
           intentos: int = 3, **ajustes) -> tuple[np.ndarray, int]:
    """Texto -> audio con tu voz. Devuelve (audio, sr)."""
    modelo = cargar_modelo()
    opciones = {**ESTILOS.get(estilo, ESTILOS["natural"]), **ajustes}
    limite = 2.0 + 0.075 * len(texto)
    audio = None
    for intento in range(intentos):
        if torch is not None:
            torch.manual_seed(semilla + intento)
        onda = modelo.generate(texto, language_id=idioma,
                               audio_prompt_path=str(voz_referencia), **opciones)
        audio = onda.squeeze(0).cpu().numpy().astype(np.float32)
        if len(audio) / modelo.sr <= limite or idioma in {"zh", "ja", "ko"}:
            break
    assert audio is not None
    if guardar is not None:
        guardar = Path(guardar)
        guardar.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(guardar), audio, modelo.sr)
    return audio, int(modelo.sr)


def partir_en_frases(texto: str) -> list[str]:
    texto = " ".join(texto.split())
    frases = re.split(r"(?<=[.!?…])\s+", texto)
    frases = [f for f in frases if f]
    unidas: list[str] = []
    for frase in frases:
        if unidas and len(unidas[-1]) < 40:
            unidas[-1] += " " + frase
        else:
            unidas.append(frase)
    if len(unidas) > 1 and len(unidas[-1]) < 40:
        unidas[-2:] = [unidas[-2] + " " + unidas[-1]]
    return unidas


def leer_guion(guion: list[tuple[str, str]], voz_referencia: str | Path,
               pausa: float = 0.35, guardar: str | Path | None = None,
               idioma: str = "es", semilla: int = 7) -> tuple[np.ndarray, int]:
    """Cada frase con su estilo, unidas con `pausa` s de silencio."""
    modelo = cargar_modelo()
    silencio = np.zeros(int(pausa * modelo.sr), dtype=np.float32)
    trozos: list[np.ndarray] = []
    for texto, estilo in guion:
        audio, _ = hablar(texto, voz_referencia, estilo=estilo, idioma=idioma, semilla=semilla)
        trozos += [audio, silencio]
    final = np.concatenate(trozos[:-1]).astype(np.float32) if trozos else np.zeros(0, dtype=np.float32)
    if guardar is not None:
        sf.write(str(guardar), final, modelo.sr)
    return final, int(modelo.sr)


def leer(texto: str, voz_referencia: str | Path, estilo: str = "expresivo",
         pausa: float = 0.35, guardar: str | Path | None = None,
         idioma: str = "es", semilla: int = 7) -> tuple[np.ndarray, int]:
    return leer_guion([(f, estilo) for f in partir_en_frases(texto)],
                      voz_referencia, pausa=pausa, guardar=guardar,
                      idioma=idioma, semilla=semilla)
