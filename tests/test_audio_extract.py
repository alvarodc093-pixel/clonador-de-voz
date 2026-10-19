import numpy as np
import pytest
import soundfile as sf

from src import audio_extract as ae
from src.audio_extract import (incorporar_archivo_local, procesar_origen,
                               extraer_audio, herramientas_ok)


def _wav(ruta, audio, sr):
    sf.write(str(ruta), np.asarray(audio, dtype="float32"), sr)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    videos = tmp_path / "videos"
    audios = tmp_path / "audios"
    videos.mkdir()
    audios.mkdir()
    monkeypatch.setattr(ae, "DIR_VIDEOS", videos)
    monkeypatch.setattr(ae, "DIR_AUDIOS", audios)
    monkeypatch.setattr(ae, "_buscar_ffmpeg", lambda nombre: None)
    return videos, audios


def test_incorporar_wav_normaliza_mono_16k(dirs, tmp_path):
    videos, audios = dirs
    origen = tmp_path / "voz.wav"
    # 1 s estéreo a 44.1 kHz
    _wav(origen, np.zeros((44100, 2), dtype="float32"), 44100)
    info = incorporar_archivo_local(origen)
    destino = audios / info["archivo_audio"]
    assert destino.exists()
    audio, sr = sf.read(str(destino), always_2d=True)
    assert sr == 16000
    assert audio.shape[1] == 1
    assert abs(audio.shape[0] - 16000) <= 2


def test_incorporar_no_reextrae_si_wav_es_reciente(dirs, tmp_path):
    videos, audios = dirs
    origen = tmp_path / "voz.wav"
    _wav(origen, np.zeros(16000, dtype="float32"), 16000)
    info1 = incorporar_archivo_local(origen)
    destino = audios / info1["archivo_audio"]
    primer_mtime = destino.stat().st_mtime
    info2 = incorporar_archivo_local(origen)
    assert info2["archivo_audio"] == info1["archivo_audio"]
    assert destino.stat().st_mtime == primer_mtime


def test_incorporar_formato_no_soportado(dirs, tmp_path):
    origen = tmp_path / "nota.txt"
    origen.write_text("hola")
    with pytest.raises(ValueError):
        incorporar_archivo_local(origen)


def test_incorporar_inexistente(dirs, tmp_path):
    with pytest.raises(FileNotFoundError):
        incorporar_archivo_local(tmp_path / "nope.wav")


def test_procesar_origen_vacio():
    with pytest.raises(ValueError):
        procesar_origen("   ")


def test_herramientas_ok_tiene_claves():
    estado = herramientas_ok()
    assert {"ffmpeg", "ffprobe", "yt-dlp"} <= set(estado)


def test_extraer_audio_sin_entrada(monkeypatch, tmp_path):
    monkeypatch.setattr(ae, "_buscar_ffmpeg", lambda nombre: "ffmpeg-falso")
    with pytest.raises(FileNotFoundError):
        extraer_audio(tmp_path / "no-existe.mp4")


def test_extraer_audio_frecuencia_invalida(monkeypatch, tmp_path):
    monkeypatch.setattr(ae, "_buscar_ffmpeg", lambda nombre: "ffmpeg-falso")
    origen = tmp_path / "x.wav"
    _wav(origen, np.zeros(100, dtype="float32"), 16000)
    with pytest.raises(ValueError):
        extraer_audio(origen, frecuencia=0)


def test_existe_video_ignora_vacios(dirs):
    videos, _ = dirs
    vacio = videos / "abc.mp4"
    vacio.write_bytes(b"")
    assert ae._existe_video(videos, "abc") is None
    bueno = videos / "abc.mp4"
    bueno.write_bytes(b"datos")
    assert ae._existe_video(videos, "abc") == bueno
