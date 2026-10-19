import numpy as np
import pytest
import soundfile as sf

from src.mix import _a_estereo, cargar_cancion_estereo, mezclar


def _escribir_wav(ruta, audio, sr):
    sf.write(str(ruta), np.asarray(audio, dtype="float32"), sr)


def test_cargar_cancion_estereo_mono_se_duplica(tmp_path):
    ruta = tmp_path / "mono.wav"
    _escribir_wav(ruta, np.zeros(4410, dtype="float32"), 44100)
    audio, sr = cargar_cancion_estereo(ruta)
    assert sr == 44100
    assert audio.shape == (2, 4410)


def test_cargar_cancion_estereo_estéreo(tmp_path):
    ruta = tmp_path / "stereo.wav"
    # soundfile escribe (muestras, canales); la función debe devolver (canales, muestras)
    _escribir_wav(ruta, np.zeros((2205, 2), dtype="float32"), 44100)
    audio, sr = cargar_cancion_estereo(ruta)
    assert audio.shape == (2, 2205)


def test_cargar_cancion_estereo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_cancion_estereo(tmp_path / "nope.wav")


def test_a_estereo_tres_canales():
    audio = np.zeros((3, 100), dtype="float32")
    assert _a_estereo(audio).shape == (2, 100)


def test_mezclar_forma_y_suma():
    voz = np.ones(44100, dtype="float32") * 0.1
    base = np.ones((2, 44100), dtype="float32") * 0.2
    out = mezclar(voz, base)
    assert out.shape == (2, 44100)
    assert np.allclose(out[0, 2000], 0.3, atol=1e-4)


def test_mezclar_recorta_a_la_pista_mas_corta():
    voz = np.ones(1000, dtype="float32")
    base = np.ones((2, 500), dtype="float32")
    assert mezclar(voz, base).shape == (2, 500)


def test_mezclar_voz_vacia():
    with pytest.raises(ValueError):
        mezclar(np.zeros(0, dtype="float32"), np.zeros((2, 100), dtype="float32"))


def test_mezclar_base_vacia():
    with pytest.raises(ValueError):
        mezclar(np.zeros(100, dtype="float32"), np.zeros((2, 0), dtype="float32"))


def test_mezclar_evita_saturacion():
    voz = np.ones(1000, dtype="float32")
    base = np.ones((2, 1000), dtype="float32")
    out = mezclar(voz, base)
    assert float(np.abs(out).max()) <= 0.981


def test_mezclar_remuestrea_voz():
    voz = np.zeros(24000, dtype="float32")  # 1 s a 24 kHz
    base = np.zeros((2, 44100), dtype="float32")
    assert mezclar(voz, base, sr=44100, sr_voz=24000).shape == (2, 44100)


def test_mezclar_guarda_archivo(tmp_path):
    destino = tmp_path / "sub" / "mezcla.wav"
    mezclar(np.zeros(1000, dtype="float32"),
            np.zeros((2, 1000), dtype="float32"), guardar=destino)
    assert destino.exists()
    audio, sr = sf.read(str(destino), always_2d=True)
    assert audio.shape == (1000, 2)


def test_mezclar_nivel_referencia():
    voz = np.ones(1000, dtype="float32") * 0.01
    base = np.zeros((2, 1000), dtype="float32")
    ref = np.ones(1000, dtype="float32") * 0.5
    out = mezclar(voz, base, nivel_referencia=ref)
    # El RMS de salida debe acercarse al de la referencia.
    rms = float(np.sqrt((out[0] ** 2).mean()))
    assert rms > 0.3


def test_mezclar_volumen_no_finito():
    with pytest.raises(ValueError):
        mezclar(np.zeros(100, dtype="float32"),
                np.zeros((2, 100), dtype="float32"), volumen_voz=float("nan"))
