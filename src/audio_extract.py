"""Paso 1 del bot: descargar vídeo y extraer solo la voz en WAV.

Fiel a la guía `Archivos/01_descargar_video_y_extraer_audio`:
yt-dlp descarga (480p basta, la imagen da igual) y ffmpeg extrae
`-vn -ac 1 -ar 16000 -c:a pcm_s16le`.

También acepta archivos locales (mp4/mp3/m4a/wav) sin descargar nada.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import DIR_AUDIOS, DIR_VIDEOS, FICHA_VIDEOS, FS_VOZ


def guardar_fichas(fichas: list[dict]) -> Path:
    """Escribe el índice data/videos.json que usan los pasos siguientes (guía 01 §5)."""
    import json

    FICHA_VIDEOS.write_text(json.dumps(fichas, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return FICHA_VIDEOS


def herramientas_ok() -> dict[str, str | None]:
    """Devuelve rutas de ffmpeg/ffprobe/yt-dlp o None si faltan."""
    estado = {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
    }
    try:
        import yt_dlp  # noqa: F401
        from yt_dlp.version import __version__

        estado["yt-dlp"] = f"python ({__version__})"
    except Exception:
        estado["yt-dlp"] = shutil.which("yt-dlp")
    return estado


def ficha_youtube(url: str) -> dict:
    """Ficha del vídeo SIN descargarlo (título, canal, duración...)."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id"),
        "titulo": info.get("title"),
        "canal": info.get("channel"),
        "duracion_s": info.get("duration"),
        "licencia": info.get("license") or "licencia estándar de YouTube",
        "origen": url,
    }


def _existe_video(destino: Path, video_id: str) -> Path | None:
    candidatos = [p for p in destino.glob(f"{video_id}.*")
                  if p.suffix not in {".part", ".ytdl", ".json"}]
    return candidatos[0] if candidatos else None


def descargar_video(url: str, altura_max: int = 480, destino: Path = DIR_VIDEOS,
                    ficha: dict | None = None) -> Path:
    """Descarga el vídeo y devuelve la ruta del mp4. Si ya está, no lo repite."""
    import yt_dlp

    destino.mkdir(parents=True, exist_ok=True)
    ficha = ficha or ficha_youtube(url)  # una sola llamada de red en todo el flujo
    existente = _existe_video(destino, ficha["id"])
    if existente is not None:
        return existente
    opciones = {
        "format": f"bv*[height<={altura_max}]+ba/b[height<={altura_max}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(destino / "%(id)s.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opciones) as ydl:
        ydl.download([url])
    existente = _existe_video(destino, ficha["id"])
    if existente is None:
        raise FileNotFoundError(f"No se encontró el vídeo descargado {ficha['id']}")
    return existente


def extraer_audio(ruta_video: str | Path, ruta_audio: str | Path | None = None,
                  frecuencia: int = FS_VOZ) -> Path:
    """Extrae la pista de sonido a WAV mono de `frecuencia` Hz."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Falta ffmpeg. Windows: `winget install ffmpeg` · "
            "macOS: `brew install ffmpeg` · Linux: `sudo apt install ffmpeg`.")
    ruta_video = Path(ruta_video)
    if ruta_audio is None:
        DIR_AUDIOS.mkdir(parents=True, exist_ok=True)
        ruta_audio = DIR_AUDIOS / f"{ruta_video.stem}.wav"
    ruta_audio = Path(ruta_audio)
    orden = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(ruta_video),
             "-vn", "-ac", "1", "-ar", str(frecuencia), "-c:a", "pcm_s16le", str(ruta_audio)]
    subprocess.run(orden, check=True)
    return ruta_audio


def incorporar_archivo_local(ruta: str | Path) -> dict:
    """Copia un archivo propio a data/videos y normaliza su wav a mono 16 kHz."""
    ruta = Path(ruta).expanduser()
    if not ruta.exists():
        raise FileNotFoundError(ruta)
    destino = DIR_VIDEOS / ruta.name
    if not destino.exists() and destino.resolve() != ruta.resolve():
        shutil.copy(ruta, destino)
    wav = DIR_AUDIOS / f"{ruta.stem}.wav"
    if not wav.exists():
        try:
            extraer_audio(destino, wav)  # normaliza: mono + 16 kHz
        except RuntimeError:
            if ruta.suffix.lower() != ".wav":
                raise
            shutil.copy(destino, wav)  # sin ffmpeg: solo vale si ya es wav
    return {"id": ruta.stem, "titulo": ruta.stem, "archivo_video": destino.name,
            "archivo_audio": wav.name, "origen": str(ruta)}


def procesar_origen(origen: str | Path, altura_max: int = 480) -> dict:
    """Un solo punto de entrada: URL de YouTube o ruta local -> dict con wav."""
    if isinstance(origen, Path) or (isinstance(origen, str) and not origen.startswith("http")):
        return incorporar_archivo_local(origen)
    ficha = ficha_youtube(str(origen))  # una sola llamada de red
    video = descargar_video(str(origen), altura_max=altura_max, ficha=ficha)
    # id = nombre sin extensión (yt-dlp usa %(id)s como nombre)
    wav = DIR_AUDIOS / f"{video.stem}.wav"
    if not wav.exists():
        extraer_audio(video, wav)
    return {**ficha, "archivo_video": video.name, "archivo_audio": wav.name}
