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

EXTENSIONES_MEDIA = {".mp4", ".m4v", ".mkv", ".webm", ".mov",
                     ".mp3", ".m4a", ".aac", ".ogg", ".wav", ".flac"}


def guardar_fichas(fichas: list[dict]) -> Path:
    """Escribe el índice data/videos.json que usan los pasos siguientes (guía 01 §5)."""
    import json

    FICHA_VIDEOS.parent.mkdir(parents=True, exist_ok=True)
    FICHA_VIDEOS.write_text(json.dumps(list(fichas), ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return FICHA_VIDEOS


def _buscar_ffmpeg(nombre: str) -> str | None:
    """Busca ffmpeg/ffprobe en PATH y, si falla, en ubicaciones típicas de Windows.

    Necesario porque `winget install Gyan.FFmpeg` añade el PATH solo para
    terminales nuevas: un Streamlit ya abierto no lo ve con shutil.which.
    """
    ruta = shutil.which(nombre)
    if ruta:
        return ruta
    # Nombres en minúsculas: el glob de pathlib distingue mayúsculas incluso en Windows
    exe = f"{nombre}.exe" if nombre in {"ffmpeg", "ffprobe"} else nombre
    candidatos: list[Path] = []
    # Paquete winget Gyan.FFmpeg (ruta versionada, p. ej. ffmpeg-9.0.2-full_build/bin)
    base_winget = (Path.home() / "AppData" / "Local" / "Microsoft"
                   / "WinGet" / "Packages")
    if base_winget.exists():
        candidatos += sorted(base_winget.glob(f"Gyan.FFmpeg*/ffmpeg-*/bin/{exe}"))
        candidatos += sorted(base_winget.glob(f"Gyan.FFmpeg*/bin/{exe}"))
    candidatos += [
        Path(r"C:\Program Files\FFmpeg\bin") / exe,
        Path(r"C:\ffmpeg\bin") / exe,
    ]
    for c in candidatos:
        if c.exists():
            return str(c)
    return None


def herramientas_ok() -> dict[str, str | None]:
    """Devuelve rutas de ffmpeg/ffprobe/yt-dlp o None si faltan."""
    estado = {
        "ffmpeg": _buscar_ffmpeg("ffmpeg"),
        "ffprobe": _buscar_ffmpeg("ffprobe"),
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

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "noplaylist": True}) as ydl:
            info = ydl.extract_info(str(url), download=False)
    except Exception as e:  # red, 403, vídeo privado, etc.
        raise RuntimeError(f"No se pudo leer el vídeo {url}: {e}") from e
    if not info or not info.get("id"):
        raise RuntimeError(f"yt-dlp no devolvió datos del vídeo {url}.")
    return {
        "id": info.get("id"),
        "titulo": info.get("title") or info.get("id"),
        "canal": info.get("channel"),
        "duracion_s": info.get("duration"),
        "licencia": info.get("license") or "licencia estándar de YouTube",
        "origen": url,
    }


def _existe_video(destino: Path, video_id: str) -> Path | None:
    candidatos = [p for p in destino.glob(f"{video_id}.*")
                  if p.suffix.lower() in EXTENSIONES_MEDIA and p.stat().st_size > 0]
    return candidatos[0] if candidatos else None


def descargar_video(url: str, altura_max: int = 480, destino: Path = DIR_VIDEOS,
                    ficha: dict | None = None) -> Path:
    """Descarga el vídeo y devuelve la ruta del mp4. Si ya está, no lo repite."""
    import yt_dlp

    destino.mkdir(parents=True, exist_ok=True)
    altura_max = int(altura_max)
    if altura_max <= 0:
        raise ValueError("altura_max debe ser positiva.")
    ficha = ficha or ficha_youtube(url)  # una sola llamada de red en todo el flujo
    if not ficha.get("id"):
        raise RuntimeError("La ficha del vídeo no tiene id; no se puede descargar.")
    existente = _existe_video(destino, ficha["id"])
    if existente is not None:
        return existente
    opciones = {
        "format": f"bv*[height<={altura_max}]+ba/b[height<={altura_max}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(destino / "%(id)s.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True, "noprogress": True,
    }
    # yt-dlp lanza ffmpeg como subproceso: si el PATH del proceso es viejo
    # (Streamlit abierto antes de instalar ffmpeg), hay que darle la ruta.
    bin_ffmpeg = _buscar_ffmpeg("ffmpeg")
    if bin_ffmpeg is not None:
        opciones["ffmpeg_location"] = str(Path(bin_ffmpeg).parent)
    try:
        with yt_dlp.YoutubeDL(opciones) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(f"Falló la descarga de {url}: {e}") from e
    existente = _existe_video(destino, ficha["id"])
    if existente is None:
        raise FileNotFoundError(f"No se encontró el vídeo descargado {ficha['id']}")
    return existente


def extraer_audio(ruta_video: str | Path, ruta_audio: str | Path | None = None,
                  frecuencia: int = FS_VOZ) -> Path:
    """Extrae la pista de sonido a WAV mono de `frecuencia` Hz."""
    bin_ffmpeg = _buscar_ffmpeg("ffmpeg")
    if bin_ffmpeg is None:
        raise RuntimeError(
            "Falta ffmpeg. Windows: `winget install ffmpeg` · "
            "macOS: `brew install ffmpeg` · Linux: `sudo apt install ffmpeg`.")
    ruta_video = Path(ruta_video)
    if not ruta_video.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {ruta_video}")
    frecuencia = int(frecuencia)
    if frecuencia <= 0:
        raise ValueError("La frecuencia de salida debe ser positiva.")
    if ruta_audio is None:
        DIR_AUDIOS.mkdir(parents=True, exist_ok=True)
        ruta_audio = DIR_AUDIOS / f"{ruta_video.stem}.wav"
    ruta_audio = Path(ruta_audio)
    ruta_audio.parent.mkdir(parents=True, exist_ok=True)
    orden = [bin_ffmpeg, "-y", "-loglevel", "error", "-i", str(ruta_video),
             "-vn", "-ac", "1", "-ar", str(frecuencia), "-c:a", "pcm_s16le", str(ruta_audio)]
    try:
        subprocess.run(orden, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        detalle = (e.stderr or "").strip()[-500:] or "sin detalle de ffmpeg"
        raise RuntimeError(f"ffmpeg no pudo extraer el audio: {detalle}") from e
    except OSError as e:
        raise RuntimeError(f"No se pudo ejecutar ffmpeg ({bin_ffmpeg}): {e}") from e
    if not ruta_audio.exists() or ruta_audio.stat().st_size == 0:
        raise RuntimeError("ffmpeg terminó sin generar el WAV esperado.")
    return ruta_audio


def _normalizar_wav_sin_ffmpeg(origen: Path, destino: Path) -> Path:
    """Fallback sin ffmpeg: soundfile + remuestreo/mono en Python."""
    import numpy as np
    import soundfile as sf

    from .utils import guardar_wav, remuestrear

    audio, sr = sf.read(str(origen), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if int(sr) != FS_VOZ:
        audio = remuestrear(audio, sr, FS_VOZ)
    return guardar_wav(destino, audio, FS_VOZ)


def incorporar_archivo_local(ruta: str | Path) -> dict:
    """Copia un archivo propio a data/videos y normaliza su wav a mono 16 kHz."""
    ruta = Path(ruta).expanduser()
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    if ruta.suffix.lower() not in EXTENSIONES_MEDIA:
        raise ValueError(
            f"Formato no soportado: {ruta.suffix or '(sin extensión)'}. "
            f"Usa: {', '.join(sorted(EXTENSIONES_MEDIA))}.")
    destino = DIR_VIDEOS / ruta.name
    if destino.resolve() != ruta.resolve() and not destino.exists():
        shutil.copy(ruta, destino)
    wav = DIR_AUDIOS / f"{ruta.stem}.wav"
    # Reextrae si el wav no existe o es más antiguo que el archivo de origen.
    if wav.exists() and wav.stat().st_mtime >= ruta.stat().st_mtime:
        return {"id": ruta.stem, "titulo": ruta.stem, "archivo_video": destino.name,
                "archivo_audio": wav.name, "origen": str(ruta)}
    try:
        extraer_audio(destino, wav)  # normaliza: mono + 16 kHz
    except RuntimeError:
        if ruta.suffix.lower() != ".wav":
            raise
        # Sin ffmpeg: normalizamos el WAV con soundfile (mono 16 kHz).
        _normalizar_wav_sin_ffmpeg(destino, wav)
    return {"id": ruta.stem, "titulo": ruta.stem, "archivo_video": destino.name,
            "archivo_audio": wav.name, "origen": str(ruta)}


def procesar_origen(origen: str | Path, altura_max: int = 480) -> dict:
    """Un solo punto de entrada: URL de YouTube o ruta local -> dict con wav."""
    if isinstance(origen, Path):
        return incorporar_archivo_local(origen)
    origen = str(origen).strip()
    if not origen:
        raise ValueError("El origen está vacío: pasa una URL o una ruta de archivo.")
    if not origen.startswith(("http://", "https://")):
        return incorporar_archivo_local(origen)
    ficha = ficha_youtube(origen)  # una sola llamada de red
    video = descargar_video(origen, altura_max=altura_max, ficha=ficha)
    # id = nombre sin extensión (yt-dlp usa %(id)s como nombre)
    wav = DIR_AUDIOS / f"{video.stem}.wav"
    if not wav.exists() or wav.stat().st_mtime < video.stat().st_mtime:
        extraer_audio(video, wav)
    return {**ficha, "archivo_video": video.name, "archivo_audio": wav.name}
