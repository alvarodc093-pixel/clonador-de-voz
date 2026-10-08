"""Configuración central del proyecto.

Rutas de data/ + valores de la guía (cuaderno 01: PERSONAJE, DESCRIPCION,
VIDEOS) leídos del `.env` de la raíz, con valores seguros por defecto.
Todo el bot (scripts/bot.py) y la app (app.py) importan de aquí: no hay
rutas ni ajustes hardcodeados en el resto del código.

El `.env` real no se sube a git; `.env.example` es la plantilla.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass  # sin python-dotenv se usan los valores por defecto


def _get(nombre: str, defecto: str = "") -> str:
    return os.getenv(nombre, defecto).strip()


def _get_int(nombre: str, defecto: int) -> int:
    try:
        return int(_get(nombre, str(defecto)))
    except ValueError:
        return defecto

DATA = ROOT / "data"
DIR_VIDEOS = DATA / "videos"
DIR_AUDIOS = DATA / "audios"          # wav extraídos de vídeos (mono 16 kHz)
DIR_REFERENCIAS = DATA / "referencias"  # mi_voz.wav y otras voces de destino
DIR_SEPARACIONES = DATA / "separaciones"  # voz sola + instrumental (Demucs)
DIR_CLONADOS = DATA / "clonados"      # salidas Chatterbox / Seed-VC
DIR_MEZCLAS = DATA / "mezclas"        # mezcla final voz + base musical

for carpeta in (DIR_VIDEOS, DIR_AUDIOS, DIR_REFERENCIAS, DIR_SEPARACIONES, DIR_CLONADOS, DIR_MEZCLAS):
    carpeta.mkdir(parents=True, exist_ok=True)

# Frecuencias de trabajo (según las guías de Archivos/)
FS_VOZ = _get_int("FS_VOZ", 16_000)       # extracción de vídeos / modelos de voz
FS_MUSICA = _get_int("FS_MUSICA", 44_100)  # separación y mezcla musical (calidad CD)
FS_CHATTERBOX = 24_000  # lo que genera Chatterbox

# Guía 01: personaje y lista de vídeos (se editan en el .env)
PERSONAJE = _get("PERSONAJE", "Mi personaje")
DESCRIPCION = _get("DESCRIPCION", "")
VIDEOS = [v.strip() for v in _get("VIDEOS", "").split(",") if v.strip()]
ALTURA_MAX = _get_int("ALTURA_MAX", 480)

# Índice que usan los pasos siguientes (como datos/videos.json de la guía)
FICHA_VIDEOS = DATA / "videos.json"

# Referencia de voz por defecto
MI_VOZ = DIR_REFERENCIAS / "mi_voz.wav"

TEXTO_EJEMPLO_GRABACION = (
    "Hola, esta es mi voz. Te cuento lo que me pasó ayer por la tarde: "
    "salí a dar un paseo, empezó a llover de repente y acabé merendando "
    "chocolate caliente en una cafetería pequeña. ¿Y sabes qué? "
    "Fue el mejor rato de toda la semana."
)
