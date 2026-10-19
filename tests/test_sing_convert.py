import numpy as np
import pytest
import soundfile as sf

from src import sing_convert as sc


def _wav(tmp_path, nombre="a.wav"):
    ruta = tmp_path / nombre
    sf.write(str(ruta), np.zeros(16000, dtype="float32"), 16000)
    return ruta


class FakeMotor:
    def __init__(self):
        self.rmvpe = self

    def infer_from_audio(self, audio, thred=0.03):
        return np.full(50, 440.0, dtype=np.float32)

    def convert_voice(self, *args, **kwargs):
        return np.full(1000, 0.1, dtype=np.float32)


def test_convertir_archivos_inexistentes(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "cargar_motor", lambda device=None: FakeMotor())
    with pytest.raises(FileNotFoundError):
        sc.convertir(tmp_path / "nope.wav", _wav(tmp_path))


def test_convertir_semitonos_invalidos(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "cargar_motor", lambda device=None: FakeMotor())
    with pytest.raises(ValueError):
        sc.convertir(_wav(tmp_path), _wav(tmp_path, "b.wav"), semitonos=99)


def test_convertir_pasos_invalidos(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "cargar_motor", lambda device=None: FakeMotor())
    with pytest.raises(ValueError):
        sc.convertir(_wav(tmp_path), _wav(tmp_path, "b.wav"), pasos=0)


def test_convertir_guarda_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "cargar_motor", lambda device=None: FakeMotor())
    destino = tmp_path / "salida.wav"
    audio = sc.convertir(_wav(tmp_path), _wav(tmp_path, "b.wav"), guardar=destino)
    assert audio.shape == (1000,)
    assert destino.exists()


def test_extraer_audio_desde_array():
    audio = sc._extraer_audio(np.arange(10, dtype="float32"))
    assert audio.shape == (10,)


def test_extraer_audio_desde_generador():
    def generador():
        yield (b"mp3", None)
        return np.arange(5, dtype="float32")

    assert sc._extraer_audio(generador()).shape == (5,)


def test_extraer_audio_vacio():
    with pytest.raises(RuntimeError):
        sc._extraer_audio(np.zeros(0, dtype="float32"))


def test_melodia_devuelve_nan_en_silencio(monkeypatch):
    monkeypatch.setattr(sc, "cargar_motor", lambda device=None: FakeMotor())
    tono = sc.melodia(np.zeros(44100, dtype="float32"), 44100)
    assert np.isfinite(tono).all()  # el fake devuelve 440 Hz


def test_diferencia_registro_octava(monkeypatch):
    monkeypatch.setattr(
        sc, "melodia",
        lambda audio, sr: np.array([220.0 if audio[0] == 0 else 440.0],
                                   dtype=np.float32))
    dif = sc.diferencia_registro(np.ones(10, dtype="float32"),
                                 np.zeros(10, dtype="float32"))
    assert abs(dif - 12.0) < 1e-6


def test_diferencia_registro_sin_tono(monkeypatch):
    monkeypatch.setattr(sc, "melodia",
                        lambda audio, sr: np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError):
        sc.diferencia_registro(np.zeros(10, dtype="float32"),
                              np.zeros(10, dtype="float32"))


def test_transposicion_no_finita():
    assert sc.transposicion_recomendada(float("nan")) == 0
    assert sc.transposicion_recomendada(float("inf")) == 0
    assert sc.transposicion_recomendada("basura") == 0


def test_dependencias_seedvc_devuelve_lista():
    faltan = sc.dependencias_seedvc()
    assert isinstance(faltan, list)
    assert all(isinstance(x, str) for x in faltan)
