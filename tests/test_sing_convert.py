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


def test_parchear_bigvgan_sin_modulos_no_lanza():
    # sin seed-vc en sys.path el parche debe ser un no-op silencioso
    sc._parchear_bigvgan_hub()


def test_parchear_bigvgan_compatibiliza_hub_nuevo(monkeypatch):
    import inspect
    import sys
    import types

    from huggingface_hub import hf_hub_download

    params = set(inspect.signature(hf_hub_download).parameters)
    if {"proxies", "resume_download"} <= params:
        pytest.skip("huggingface_hub antiguo: el parche no hace falta")

    llamadas = {}

    def fake_fhd(repo_id, filename, **kwargs):
        # misma forma que hub 1.x: sin proxies ni resume_download
        llamadas["fhd_kwargs"] = {"repo_id": repo_id, "filename": filename,
                                  **kwargs}
        return "modelo.bin"

    class FakeBigVGAN:
        @classmethod
        def _from_pretrained(cls, *, model_id, proxies, resume_download, **kwargs):
            # como en seed-vc: proxies/resume_download son obligatorios
            # sin defaults, y el hub 1.x no los envia.
            llamadas["proxies"] = proxies
            llamadas["resume_download"] = resume_download
            mod = sys.modules["modules.bigvgan.bigvgan"]
            llamadas["fhd"] = mod.hf_hub_download(
                repo_id=model_id, filename="generator.pth",
                proxies=proxies, resume_download=resume_download)
            return "cargado"

    bg = types.ModuleType("modules.bigvgan.bigvgan")
    bg.hf_hub_download = fake_fhd
    bg.BigVGAN = FakeBigVGAN
    for nombre, mod in {
        "modules": types.ModuleType("modules"),
        "modules.bigvgan": types.ModuleType("modules.bigvgan"),
        "modules.bigvgan.bigvgan": bg,
    }.items():
        monkeypatch.setitem(sys.modules, nombre, mod)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_fhd)

    sc._parchear_bigvgan_hub()

    # _from_pretrained ya no exige proxies/resume_download (tiene defaults)
    assert FakeBigVGAN._from_pretrained(model_id="x") == "cargado"
    assert llamadas["proxies"] is None
    assert llamadas["resume_download"] is False
    # y el wrapper de hf_hub_download filtra los kwargs obsoletos
    assert "proxies" not in llamadas["fhd_kwargs"]
    assert "resume_download" not in llamadas["fhd_kwargs"]
    assert llamadas["fhd_kwargs"]["repo_id"] == "x"
    assert llamadas["fhd_kwargs"]["filename"] == "generator.pth"
