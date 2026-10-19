from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_carga_sin_errores():
    at = _app()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Clonador de voz"
    assert len(at.tabs) == 5


def test_app_muestra_estado_del_sistema():
    at = _app()
    textos = " ".join(m.value for m in at.sidebar.markdown)
    assert "ffmpeg" in textos
    assert "Seed-VC" in textos


def test_extraer_sin_origen_avisa():
    at = _app()
    botones = [b for b in at.button if b.label == "Extraer audio"]
    assert botones, "No se encontró el botón de extraer"
    botones[0].click().run()
    assert not at.exception
    assert any("Pega un enlace" in w.value for w in at.warning)


def test_no_hay_excepcion_al_arrancar():
    at = _app()
    assert at.exception == []
