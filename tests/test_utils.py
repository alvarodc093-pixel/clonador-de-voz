import numpy as np
import pytest

from src import config as cfg
from src.clone_tts import partir_en_frases
from src.mix import mezclar
from src.sing_convert import transposicion_recomendada
from src.utils import (fundido_cruzado_simple, normalizar_pico, remuestrear,
                       revisar_grabacion)


def test_revision_ok():
    sr = 16000
    audio = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr * 10) / sr)).astype("float32")
    r = revisar_grabacion(audio, sr)
    assert r["ok"]


def test_normalizar_pico():
    audio = np.array([0.1, -0.5, 0.25], dtype="float32")
    out = normalizar_pico(audio, 0.89)
    assert abs(float(abs(out).max()) - 0.89) < 1e-5


def test_remuestrear_misma_frecuencia():
    audio = np.arange(100, dtype="float32")
    assert len(remuestrear(audio, 16000, 16000)) == 100


def test_remuestrear_cambia_longitud():
    audio = np.zeros(16000, dtype="float32")
    out = remuestrear(audio, 16000, 44100)
    assert abs(len(out) - 44100) <= 1


def test_mezclar_voz_otro_sr():
    # Voz Chatterbox a 24 kHz sobre base a 44.1 kHz: debe remuestrearse, no sumarse tal cual
    voz = np.zeros(24000, dtype="float32")  # 1 s a 24 kHz
    base = np.zeros((2, 44100), dtype="float32")
    mezcla = mezclar(voz, base, sr=44100, sr_voz=24000)
    assert mezcla.shape == (2, 44100)


def test_partir_frases_une_cortas():
    frases = partir_en_frases("Sí. Buenos días a todos, ¿cómo estáis hoy?")
    assert len(frases) == 1 and frases[0].startswith("Sí.")


def test_transposicion_regla_octava():
    assert transposicion_recomendada(11) == -12
    assert transposicion_recomendada(-9) == 12
    assert transposicion_recomendada(3) == 0


def test_config_env_por_defecto():
    assert cfg.FS_VOZ == 16000
    assert cfg.FS_MUSICA == 44100
    assert isinstance(cfg.VIDEOS, list)


def test_revision_archivo_vacio_no_falla():
    r = revisar_grabacion(np.zeros(0, dtype="float32"), 16000)
    assert r["duracion"] == 0.0
    assert not r["ok"]


def test_revision_con_sr_invalido():
    r = revisar_grabacion(np.zeros(1000, dtype="float32"), 0)
    assert r["duracion"] == 0.0


def test_revision_saturada_no_ok():
    audio = np.ones(16000 * 10, dtype="float32")
    assert not revisar_grabacion(audio, 16000)["ok"]


def test_normalizar_pico_vacio():
    out = normalizar_pico(np.zeros(0, dtype="float32"))
    assert out.size == 0


def test_normalizar_pico_silencio():
    audio = np.zeros(100, dtype="float32")
    assert np.array_equal(normalizar_pico(audio), audio)


def test_normalizar_pico_quita_nan():
    out = normalizar_pico(np.array([np.nan, 0.5, -0.5], dtype="float32"))
    assert np.isfinite(out).all()


def test_remuestrear_vacio():
    assert remuestrear(np.zeros(0, dtype="float32"), 16000, 44100).size == 0


def test_remuestrear_sr_invalido():
    with pytest.raises(ValueError):
        remuestrear(np.zeros(10, dtype="float32"), 0, 44100)


def test_fundido_no_sobreescribe_muestras():
    audio = np.ones(20, dtype="float32")
    out = fundido_cruzado_simple(audio, sr=1000, ms=1000)  # pide 1000 muestras
    assert out.shape == audio.shape
    assert out[0] <= out[len(out) // 2]  # fade-in al inicio
    assert out[-1] <= out[len(out) // 2]  # fade-out al final


def test_fundido_audio_corto_no_falla():
    assert fundido_cruzado_simple(np.ones(1, dtype="float32"), 16000).shape == (1,)
