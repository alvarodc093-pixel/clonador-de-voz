import numpy as np
import pytest
import soundfile as sf

from src import clone_tts


def _ref(tmp_path):
    ruta = tmp_path / "ref.wav"
    sf.write(str(ruta), np.zeros(16000, dtype="float32"), 16000)
    return ruta


class FakeModel:
    sr = 24000

    def __init__(self, duraciones=None):
        self.duraciones = list(duraciones or [])
        self.calls = 0

    def generate(self, text, language_id=None, audio_prompt_path=None, **kw):
        import torch

        self.calls += 1
        if self.duraciones:
            n = self.duraciones.pop(0)
        else:
            n = 2400
        return torch.zeros(n, dtype=torch.float32)


def test_partir_en_frases_vacio():
    assert clone_tts.partir_en_frases("   ") == []


def test_partir_en_frases_separa():
    frases = clone_tts.partir_en_frases(
        "Una frase bastante larga para que no se una con la siguiente. "
        "Y otra frase igualmente larga que tampoco debería fusionarse.")
    assert len(frases) == 2


def test_hablar_texto_vacio(tmp_path):
    with pytest.raises(ValueError):
        clone_tts.hablar("", _ref(tmp_path))


def test_hablar_referencia_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        clone_tts.hablar("Hola", tmp_path / "nope.wav")


def test_hablar_idioma_no_soportado(tmp_path):
    with pytest.raises(ValueError):
        clone_tts.hablar("Hola", _ref(tmp_path), idioma="xx")


def test_hablar_estilo_no_soportado(tmp_path):
    with pytest.raises(ValueError):
        clone_tts.hablar("Hola", _ref(tmp_path), estilo="inventado")


def test_idiomas_soportados_completos():
    assert len(clone_tts.IDIOMAS_SOPORTADOS) >= 20
    assert set(clone_tts.IDIOMAS_SOPORTADOS) == set(clone_tts.IDIOMAS)


def test_hablar_genera_y_guarda(tmp_path, monkeypatch):
    monkeypatch.setattr(clone_tts, "cargar_modelo", lambda device=None: FakeModel())
    destino = tmp_path / "out.wav"
    audio, sr = clone_tts.hablar("Hola, mundo.", _ref(tmp_path), guardar=destino)
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.size == 2400
    assert destino.exists()


def test_hablar_reintenta_si_es_demasiado_largo(tmp_path, monkeypatch):
    modelo = FakeModel(duraciones=[24000 * 10, 2400])  # 10 s y luego 0.1 s
    monkeypatch.setattr(clone_tts, "cargar_modelo", lambda device=None: modelo)
    audio, _ = clone_tts.hablar("Hola", _ref(tmp_path))
    assert modelo.calls == 2
    assert audio.size == 2400


def test_leer_une_frases(tmp_path, monkeypatch):
    monkeypatch.setattr(clone_tts, "cargar_modelo", lambda device=None: FakeModel())
    audio, sr = clone_tts.leer("Buenos días. ¿Cómo estás?",
                               _ref(tmp_path), pausa=0.05)
    assert sr == 24000
    assert audio.size > 0


def test_leer_texto_vacio(tmp_path):
    with pytest.raises(ValueError):
        clone_tts.leer("   ", _ref(tmp_path))


def test_leer_guion_vacio(tmp_path):
    with pytest.raises(ValueError):
        clone_tts.leer_guion([], _ref(tmp_path))
