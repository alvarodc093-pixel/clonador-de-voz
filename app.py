"""Clonador de voz — app Streamlit.

Flujo en 5 pasos (mismas funciones que scripts/bot.py):
1. Extraer audio de un video (YouTube o archivo).
2. Registrar tu voz de referencia (10-20 s).
3. Clonar voz hablada con Chatterbox (texto -> audio).
4. Poner tu voz a cantar con Demucs + Seed-VC.
5. Mezclar voz + base musical de fondo.

Los modelos pesados se cargan solo al pulsar el boton (cacheados).
Sin GPU todo funciona salvo que la conversion tarda mas.
"""
from pathlib import Path

import numpy as np
import streamlit as st

from src import config as cfg
from src.utils import guardar_wav, leer_wav, revisar_grabacion

st.set_page_config(page_title="Clonador de voz", page_icon=":material/mic:",
                   layout="wide")

st.session_state.setdefault("audio_extraido", None)
st.session_state.setdefault("mi_voz_path", str(cfg.MI_VOZ)
                            if cfg.MI_VOZ.exists() else None)
st.session_state.setdefault("voz_clonada", None)
st.session_state.setdefault("voz_convertida", None)
st.session_state.setdefault("instrumental", None)

st.title("Clonador de voz")
st.markdown(
    "Extrae el audio de un video, registra tu voz y ponla sobre una "
    "base musical. Clona **solo tu voz o la de quien te haya dado permiso**, "
    "y avisa siempre de que el audio es generado."
)


@st.cache_data(ttl="5m")
def estado_herramientas() -> dict[str, bool]:
    from src.audio_extract import herramientas_ok
    from src.clone_tts import modelo_disponible
    from src.separate import demucs_disponible
    from src.sing_convert import seedvc_disponible

    h = herramientas_ok()
    return {
        "ffmpeg": bool(h.get("ffmpeg")),
        "yt-dlp": bool(h.get("yt-dlp")),
        "Chatterbox": modelo_disponible(),
        "Demucs": demucs_disponible(),
        "Seed-VC": seedvc_disponible(),
    }


with st.sidebar:
    st.subheader("Estado del sistema")
    for nombre, ok in estado_herramientas().items():
        st.markdown(f"{'✅' if ok else '⬜'} {nombre}")
    st.caption("⬜ = opcional. Sin el modelo, ese paso muestra como instalarlo.")
    st.divider()
    st.markdown(f"**Personaje:** {cfg.PERSONAJE}")
    if cfg.DESCRIPCION:
        st.caption(cfg.DESCRIPCION)
    if cfg.VIDEOS:
        st.caption(f"{len(cfg.VIDEOS)} video(s) en el `.env`.")
    st.markdown("Material original en `Archivos/` (cuadernos de la guía).")
    import sys as _sys
    st.caption(f"`{ _sys.executable }`")

tab_extraer, tab_voz, tab_clonar, tab_cantar, tab_mezclar = st.tabs(
    ["1. Extraer audio", "2. Mi voz", "3. Clonar voz",
     "4. Poner voz a cantar", "5. Mezcla final"])


# ---- 1. Extraer -------------------------------------------------------------
with tab_extraer:
    st.subheader("Extraer audio de un video")
    st.caption("De YouTube a WAV mono 16 kHz, listo para los modelos de voz.")
    with st.form("form_extraer"):
        opciones_url = ["Escribir URL…", *cfg.VIDEOS] if cfg.VIDEOS else ["Escribir URL…"]
        elegida = st.selectbox("Video", opciones_url,
                               help="Las opciones vienen de VIDEOS en el `.env`.")
        url = st.text_input("Enlace de YouTube",
                            placeholder="https://www.youtube.com/watch?v=...",
                            value="" if elegida == "Escribir URL…" else elegida)
        archivo = st.file_uploader("O sube un video/audio",
                                   type=["mp4", "m4v", "mp3", "m4a", "wav"])
        alturas = [480, 720, 1080]
        altura = st.selectbox("Calidad de descarga", alturas,
                              index=alturas.index(cfg.ALTURA_MAX)
                              if cfg.ALTURA_MAX in alturas else 0)
        enviar = st.form_submit_button("Extraer audio", icon=":material/download:")

    if enviar:
        from src.audio_extract import herramientas_ok, procesar_origen

        h = herramientas_ok()
        try:
            if archivo is not None:
                tmp = cfg.DIR_VIDEOS / archivo.name
                tmp.write_bytes(archivo.getbuffer())
                info = procesar_origen(tmp)
            elif url.strip():
                if not h.get("ffmpeg") or not h.get("yt-dlp"):
                    st.error("Falta ffmpeg o yt-dlp. Instala: `pip install yt-dlp` y "
                             "ffmpeg (Windows: `winget install ffmpeg`).")
                    st.stop()
                with st.spinner("Descargando y extrayendo..."):
                    info = procesar_origen(url.strip(), altura_max=int(altura))
            else:
                st.warning("Pega un enlace o sube un archivo.")
                st.stop()
            wav = cfg.DIR_AUDIOS / info["archivo_audio"]
            st.session_state["audio_extraido"] = str(wav)
            st.success(f"Audio listo: `{wav.name}`")
            st.audio(str(wav))
            audio, sr = leer_wav(wav)
            st.line_chart(audio[:: max(1, len(audio) // 2000)])
            st.caption(f"{len(audio) / sr / 60:.1f} min · {sr} Hz · mono")
        except Exception as e:
            st.error(f"No se pudo extraer: {e}")


# ---- 2. Mi voz --------------------------------------------------------------
with tab_voz:
    st.subheader("Tu voz de referencia")
    st.caption("Entre 10 y 20 segundos, sin eco ni ruido, hablando con naturalidad.")
    st.info(cfg.TEXTO_EJEMPLO_GRABACION)

    grab = st.audio_input("Graba tu voz")
    subido = st.file_uploader("O sube tu grabacion", type=["wav", "mp3", "m4a"])

    fuente = None
    if grab is not None:
        fuente = ("grabacion", grab.getvalue(), "grabacion.wav")
    elif subido is not None:
        fuente = ("archivo", subido.getvalue(), subido.name)

    if fuente is not None and st.button("Guardar como mi voz",
                                        icon=":material/save:"):
        _, datos, nombre = fuente
        tmp = cfg.DIR_REFERENCIAS / f"tmp_{nombre}"
        tmp.write_bytes(datos)
        # Normalizar a wav con soundfile; si es mp3/m4a usa librosa si esta.
        try:
            audio, sr = leer_wav(tmp) if tmp.suffix == ".wav" else (None, None)
        except Exception:
            audio, sr = None, None
        if audio is None:
            try:
                import librosa
                import soundfile as sf

                y, _ = librosa.load(str(tmp), sr=cfg.FS_VOZ, mono=True)
                if len(y) == 0:
                    raise ValueError("El archivo no contiene audio.")
                sf.write(str(cfg.MI_VOZ), y, cfg.FS_VOZ)
                audio, sr = leer_wav(cfg.MI_VOZ)
            except ImportError:
                st.error("Para mp3/m4a instala librosa o sube un WAV.")
                st.stop()
        else:
            guardar_wav(cfg.MI_VOZ, audio, sr)
        tmp.unlink(missing_ok=True)
        st.session_state["mi_voz_path"] = str(cfg.MI_VOZ)
        st.success(f"Guardada en `{cfg.MI_VOZ}`")

    if cfg.MI_VOZ.exists():
        st.audio(str(cfg.MI_VOZ))
        audio, sr = leer_wav(cfg.MI_VOZ)
        rev = revisar_grabacion(audio, sr)
        cols = st.columns(3)
        for col, ch in zip(cols, rev["cheques"]):
            col.metric(ch["nombre"], "OK" if ch["ok"] else "Revisar")
            if not ch["ok"]:
                col.caption(ch["consejo"])
        st.line_chart(audio[:: max(1, len(audio) // 2000)])
    else:
        st.warning("Aun no hay referencia. Graba o sube tu voz.")


# ---- 3. Clonar --------------------------------------------------------------
with tab_clonar:
    st.subheader("Clonar voz hablada")
    st.caption("Texto escrito a audio con tu voz (Chatterbox, 23 idiomas).")
    from src.clone_tts import ESTILOS, modelo_disponible

    if not modelo_disponible():
        st.warning("Chatterbox no instalado. Entorno de voz (Python 3.11): "
                   "`pip install -r requirements-voz.txt`")
    with st.form("form_clonar"):
        texto = st.text_area("Texto para leer", height=120,
                             value="Hola. Esta frase no la he dicho yo: "
                                   "la ha generado un modelo con mi voz.")
        col1, col2, col3 = st.columns(3)
        estilo = col1.selectbox("Estilo", list(ESTILOS), index=2)
        idioma = col2.selectbox("Idioma", ["es", "en", "fr", "it", "de", "pt"])
        semilla = col3.number_input("Semilla", value=7, step=1)
        generar = st.form_submit_button("Generar con mi voz",
                                        icon=":material/record_voice_over:")

    if generar:
        if not st.session_state.get("mi_voz_path"):
            st.error("Primero guarda tu voz en el paso 2.")
            st.stop()
        from src.clone_tts import leer

        destino = cfg.DIR_CLONADOS / "clon_hablado.wav"
        with st.spinner("Generando (la primera vez descarga ~3 GB)..."):
            try:
                audio, sr = leer(texto, st.session_state["mi_voz_path"],
                                 estilo=estilo, idioma=idioma,
                                 semilla=int(semilla), guardar=destino)
            except Exception as e:
                st.error(str(e))
                st.stop()
        st.session_state["voz_clonada"] = str(destino)
        st.success(f"{len(audio) / sr:.1f} s generados")
        st.audio(str(destino))


# ---- 4. Cantar --------------------------------------------------------------
with tab_cantar:
    st.subheader("Poner tu voz a cantar")
    st.caption("Separar voz e instrumental (Demucs) y convertir al tu timbre (Seed-VC).")
    from src.separate import demucs_disponible
    from src.sing_convert import seedvc_disponible

    if not demucs_disponible():
        st.warning("Demucs no instalado: `pip install -r requirements-canto.txt`")
    if not seedvc_disponible():
        st.warning("Falta la carpeta `seed-vc`: "
                   "`git clone https://github.com/Plachtaa/seed-vc.git`")

    cancion = st.file_uploader("Cancion base (mp3/wav con voz + acompanamiento)",
                               type=["mp3", "wav"])
    with st.form("form_cantar"):
        semitonos = st.selectbox("Transposicion",
                                 ["Auto (según registro)", 0, -12, 12], index=0,
                                 help="Auto mide tu registro y baja una octava si la "
                                      "canción queda más de 8 semitonos arriba. "
                                      "Si tu voz es más grave que la del cantante: -12.")
        convertir_btn = st.form_submit_button("Convertir a mi voz",
                                              icon=":material/music_note:")

    if convertir_btn:
        if cancion is None:
            st.warning("Sube primero la cancion base.")
            st.stop()
        if not st.session_state.get("mi_voz_path"):
            st.error("Primero guarda tu voz en el paso 2.")
            st.stop()
        try:
            from src.mix import cargar_cancion_estereo
            from src.separate import cargar_pistas, separar
            from src.sing_convert import convertir
            import soundfile as sf

            tmp = cfg.DIR_SEPARACIONES / cancion.name
            tmp.write_bytes(cancion.getbuffer())
            frag, sr = cargar_cancion_estereo(tmp)
            frag_wav = cfg.DIR_SEPARACIONES / f"{tmp.stem}_base.wav"
            sf.write(str(frag_wav), frag.T, sr)
            with st.spinner("Separando voz e instrumental (Demucs)..."):
                rutas = separar(frag_wav, cfg.DIR_SEPARACIONES / tmp.stem)
                voz_orig, inst, sr_sep = cargar_pistas(rutas)
            st.audio(str(rutas["voz"]))
            st.caption(f"Voz separada: {len(voz_orig) / sr_sep:.1f} s")
            with st.spinner("Convirtiendo a tu timbre (Seed-VC)..."):
                if semitonos == "Auto (según registro)":
                    from src.sing_convert import (diferencia_registro,
                                                  transposicion_recomendada)

                    mi_voz, sr_mia = leer_wav(st.session_state["mi_voz_path"])
                    dif = diferencia_registro(voz_orig, mi_voz, sr_sep, sr_mia)
                    semitonos = transposicion_recomendada(float(dif))
                    st.info(f"Registro: {dif:+.0f} semitonos → transponer {semitonos}.")
                voz_conv = convertir(rutas["voz"], st.session_state["mi_voz_path"],
                                     semitonos=int(semitonos),
                                     guardar=cfg.DIR_CLONADOS / "voz_convertida.wav")
            st.session_state["voz_convertida"] = str(cfg.DIR_CLONADOS / "voz_convertida.wav")
            st.session_state["instrumental"] = str(rutas["instrumental"])
            st.success("Conversion lista. Escucha la diferencia:")
            col_a, col_b = st.columns(2)
            col_a.markdown("**Original**")
            col_a.audio(str(rutas["voz"]))
            col_b.markdown("**Con tu timbre**")
            col_b.audio(str(cfg.DIR_CLONADOS / "voz_convertida.wav"))
        except Exception as e:
            st.error(f"Fallo en conversion: {e}")


# ---- 5. Mezcla --------------------------------------------------------------
with tab_mezclar:
    st.subheader("Mezcla final")
    st.caption("Tu voz clonada sobre la base musical de fondo.")
    voz_opciones = sorted(p.name for p in cfg.DIR_CLONADOS.glob("*.wav"))
    inst_sep = Path(st.session_state["instrumental"]) \
        if st.session_state.get("instrumental") and \
        Path(st.session_state["instrumental"]).exists() else None
    fuentes_base = (["Usar instrumental separado", "Subir base musical"]
                    if inst_sep else ["Subir base musical"])
    fuente_base = st.selectbox("Base musical", fuentes_base)
    base_archivo = st.file_uploader("Base musical (mp3/wav, instrumental o cancion)",
                                    type=["mp3", "wav"]) \
        if fuente_base == "Subir base musical" else None
    voz_sel = st.selectbox("Voz (de los pasos 3-4)", voz_opciones or ["(aun no hay)"])
    volumen = st.slider("Volumen de la voz", 0.3, 2.0, 1.0, 0.05)
    if st.button("Mezclar", icon=":material/graphic_eq:"):
        if not voz_opciones:
            st.warning("Genera primero una voz en los pasos 3 o 4.")
            st.stop()
        if fuente_base == "Subir base musical" and base_archivo is None:
            st.warning("Sube la base musical.")
            st.stop()
        try:
            from src.mix import cargar_cancion_estereo, mezclar

            voz, sr_voz = leer_wav(cfg.DIR_CLONADOS / voz_sel)
            if fuente_base == "Usar instrumental separado":
                base, sr = cargar_cancion_estereo(inst_sep)
                nombre_base = inst_sep.name
            else:
                tmp_base = cfg.DIR_MEZCLAS / base_archivo.name
                tmp_base.write_bytes(base_archivo.getbuffer())
                base, sr = cargar_cancion_estereo(tmp_base)
                nombre_base = base_archivo.name
            destino = cfg.DIR_MEZCLAS / "mezcla_final.wav"
            mezcla = mezclar(voz, base, sr=sr, sr_voz=sr_voz,
                             volumen_voz=float(volumen), guardar=destino)
            st.success(f"Mezcla lista ({nombre_base}): {mezcla.shape[1] / sr:.1f} s")
            st.audio(str(destino))
        except Exception as e:
            st.error(f"No se pudo mezclar: {e}")

st.divider()
st.caption("Uso responsable: solo tu voz o con permiso, canciones libres o tuyas, "
           "y avisar siempre de que la voz es generada. Seed-VC no anade marca de agua.")
