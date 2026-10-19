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
import threading
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

IDIOMAS = {
    "ar": "Árabe", "da": "Danés", "de": "Alemán", "el": "Griego",
    "en": "Inglés", "es": "Español", "fi": "Finés", "fr": "Francés",
    "he": "Hebreo", "hi": "Hindi", "it": "Italiano", "ja": "Japonés",
    "ko": "Coreano", "ms": "Malayo", "nl": "Neerlandés", "no": "Noruego",
    "pl": "Polaco", "pt": "Portugués", "ru": "Ruso", "sv": "Sueco",
    "sw": "Suajili", "tr": "Turco", "zh": "Chino",
}
IDIOMAS_SOPORTADOS = list(IDIOMAS)


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
_cerrojo = threading.Lock()


def cargar_modelo(device: str | None = None):
    """Carga Chatterbox Multilingual una sola vez (lazy). Lanza error claro si falta."""
    global _modelo
    if _modelo is not None:
        return _modelo
    with _cerrojo:
        if _modelo is not None:  # otra sesión pudo cargarlo mientras esperábamos
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
        try:
            _modelo = ChatterboxMultilingualTTS.from_pretrained(device=device)
        except Exception as e:
            raise RuntimeError(
                "No se pudo cargar Chatterbox (¿sin conexión para descargar "
                f"los ~3 GB o sin memoria suficiente?). Detalle: {e}"
            ) from e
        return _modelo


def hablar(texto: str, voz_referencia: str | Path, estilo: str = "natural",
           idioma: str = "es", semilla: int = 7, guardar: str | Path | None = None,
           intentos: int = 3, **ajustes) -> tuple[np.ndarray, int]:
    """Texto -> audio con tu voz. Devuelve (audio, sr)."""
    texto = " ".join(str(texto).split())
    if not texto:
        raise ValueError("El texto está vacío.")
    voz_referencia = Path(voz_referencia)
    if not voz_referencia.exists():
        raise FileNotFoundError(f"No existe la voz de referencia: {voz_referencia}")
    idioma = str(idioma).lower()
    if idioma not in IDIOMAS_SOPORTADOS:
        raise ValueError(
            f"Idioma no soportado: {idioma}. Usa uno de: "
            f"{', '.join(IDIOMAS_SOPORTADOS)}.")
    intentos = max(1, int(intentos))
    if estilo not in ESTILOS:
        raise ValueError(
            f"Estilo no soportado: {estilo}. Usa: {', '.join(ESTILOS)}.")
    modelo = cargar_modelo()
    if int(modelo.sr) <= 0:
        raise RuntimeError("El modelo devolvió una frecuencia de muestreo inválida.")
    opciones = {**ESTILOS[estilo], **ajustes}
    limite = 2.0 + 0.075 * len(texto)
    audio = None
    for intento in range(intentos):
        if torch is not None:
            torch.manual_seed(int(semilla) + intento)
        onda = modelo.generate(texto, language_id=idioma,
                               audio_prompt_path=str(voz_referencia), **opciones)
        audio = np.asarray(onda.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("El modelo generó audio vacío.")
        if len(audio) / modelo.sr <= limite or idioma in {"zh", "ja", "ko"}:
            break
    if audio is None:
        raise RuntimeError("No se pudo generar audio.")
    if guardar is not None:
        guardar = Path(guardar)
        guardar.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(guardar), audio, int(modelo.sr))
    return audio, int(modelo.sr)


def partir_en_frases(texto: str) -> list[str]:
    texto = " ".join(str(texto).split())
    if not texto:
        return []
    frases = [f.strip() for f in re.split(r"(?<=[.!?…])\s+", texto) if f.strip()]
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
    guion = [(str(t).strip(), e) for t, e in guion if str(t).strip()]
    if not guion:
        raise ValueError("El guion no tiene frases que leer.")
    if pausa < 0:
        raise ValueError("La pausa no puede ser negativa.")
    modelo = cargar_modelo()
    if int(modelo.sr) <= 0:
        raise RuntimeError("El modelo devolvió una frecuencia de muestreo inválida.")
    silencio = np.zeros(int(pausa * modelo.sr), dtype=np.float32)
    trozos: list[np.ndarray] = []
    for texto, estilo in guion:
        audio, _ = hablar(texto, voz_referencia, estilo=estilo, idioma=idioma,
                          semilla=semilla)
        trozos += [audio, silencio]
    final = (np.concatenate(trozos[:-1]).astype(np.float32)
             if trozos else np.zeros(0, dtype=np.float32))
    if guardar is not None:
        guardar = Path(guardar)
        guardar.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(guardar), final, int(modelo.sr))
    return final, int(modelo.sr)


def leer(texto: str, voz_referencia: str | Path, estilo: str = "expresivo",
         pausa: float = 0.35, guardar: str | Path | None = None,
         idioma: str = "es", semilla: int = 7) -> tuple[np.ndarray, int]:
    frases = partir_en_frases(texto)
    if not frases:
        raise ValueError("El texto está vacío.")
    return leer_guion([(f, estilo) for f in frases],
                      voz_referencia, pausa=pausa, guardar=guardar,
                      idioma=idioma, semilla=semilla)
