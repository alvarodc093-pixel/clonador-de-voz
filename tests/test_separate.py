from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import separate as sep


def _wav(ruta, audio, sr=44100):
    sf.write(str(ruta), np.asarray(audio, dtype="float32"), sr)
    return ruta


def test_a_mono_estéreo():
    audio = np.ones((2, 100), dtype="float32")
    assert sep._a_mono(audio).shape == (100,)


def test_a_estereo_mono():
    assert sep._a_estereo(np.ones(100, dtype="float32")).shape == (2, 100)


def test_cargar_pistas(tmp_path):
    voz = _wav(tmp_path / "vocals.wav", np.ones(1000, dtype="float32"))
    # soundfile escribe (muestras, canales)
    inst = _wav(tmp_path / "no_vocals.wav", np.ones((1000, 2), dtype="float32"))
    v, i, sr = sep.cargar_pistas({"voz": voz, "instrumental": inst})
    assert sr == 44100
    assert v.shape == (1000,)
    assert i.shape == (2, 1000)


def test_cargar_pistas_falta_clave(tmp_path):
    with pytest.raises(KeyError):
        sep.cargar_pistas({"voz": tmp_path / "x.wav"})


def test_cargar_pistas_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        sep.cargar_pistas({"voz": tmp_path / "no.wav",
                           "instrumental": tmp_path / "no2.wav"})


def test_separar_inexistente(tmp_path, monkeypatch):
    monkeypatch.setattr(sep, "demucs_disponible", lambda: True)
    with pytest.raises(FileNotFoundError):
        sep.separar(tmp_path / "no.wav", tmp_path / "out")


def test_separar_exige_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(sep, "demucs_disponible", lambda: True)
    origen = tmp_path / "cancion.mp3"
    origen.write_bytes(b"mp3")
    with pytest.raises(ValueError):
        sep.separar(origen, tmp_path / "out")


def test_separar_genera_pistas(tmp_path, monkeypatch):
    monkeypatch.setattr(sep, "demucs_disponible", lambda: True)
    cancion = _wav(tmp_path / "cancion.wav", np.zeros((1000, 2), dtype="float32"))

    class FakeProc:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        args = [str(c) for c in cmd]
        salida = Path(args[args.index("-o") + 1])
        modelo = args[args.index("-n") + 1]
        stem = Path(args[-1]).stem
        pistas = salida / modelo / stem
        pistas.mkdir(parents=True, exist_ok=True)
        (pistas / "vocals.wav").write_bytes(b"voz")
        (pistas / "no_vocals.wav").write_bytes(b"inst")
        return FakeProc()

    monkeypatch.setattr(sep.subprocess, "run", fake_run)
    rutas = sep.separar(cancion, tmp_path / "out")
    assert rutas["voz"].exists()
    assert rutas["instrumental"].exists()


def test_separar_falla_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(sep, "demucs_disponible", lambda: True)
    cancion = _wav(tmp_path / "cancion.wav", np.zeros(100, dtype="float32"))

    class FakeProc:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr(sep.subprocess, "run", lambda *a, **k: FakeProc())
    with pytest.raises(RuntimeError):
        sep.separar(cancion, tmp_path / "out")
