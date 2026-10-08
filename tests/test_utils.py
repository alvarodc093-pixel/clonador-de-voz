import numpy as np

from src import config as cfg
from src.clone_tts import partir_en_frases
from src.mix import mezclar
from src.sing_convert import transposicion_recomendada
from src.utils import normalizar_pico, remuestrear, revisar_grabacion


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
