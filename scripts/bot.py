"""Bot CLI del clonador de voz.

Une los 4 pasos de las guias en comandos sueltos (cada uno deja sus
archivos en data/ para el siguiente):

    python scripts/bot.py estado                        # que herramientas hay
    python scripts/bot.py extraer URL_O_ARCHIVO         # o sin args: usa VIDEOS del .env
    python scripts/bot.py separar data/audios/01_fragmento.wav
    python scripts/bot.py clonar --voz data/referencias/mi_voz.wav --texto "Hola, mundo"
    python scripts/bot.py convertir --fuente data/separaciones/voz.wav --ref data/referencias/mi_voz.wav
    python scripts/bot.py mezclar --voz data/clonados/voz.wav --base data/separaciones/inst.wav

La app Streamlit (app.py) llama a las mismas funciones de src/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402

VERSION = "1.1.0"


def _exigir_archivo(ruta: str | Path, etiqueta: str) -> Path:
    ruta = Path(ruta).expanduser()
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {etiqueta}: {ruta}")
    return ruta


def cmd_estado(_args) -> int:
    from src.audio_extract import herramientas_ok
    from src.clone_tts import IDIOMAS_SOPORTADOS, dispositivo, modelo_disponible
    from src.separate import demucs_disponible
    from src.sing_convert import dependencias_seedvc, seedvc_disponible

    print(f"Python: {sys.executable}")
    print(f"Dispositivo: {dispositivo()}")
    for nombre, ruta in herramientas_ok().items():
        print(f"  {'OK' if ruta else '--'}  {nombre}: {ruta or 'no encontrado'}")
    print(f"  {'OK' if modelo_disponible() else '--'}  Chatterbox (voz hablada)")
    print(f"  {'OK' if demucs_disponible() else '--'}  Demucs (separar pistas)")
    if seedvc_disponible():
        faltan = dependencias_seedvc()
        estado = "OK" if not faltan else "--"
        print(f"  {estado}  Seed-VC (voz cantada)"
              + (f" - faltan: {', '.join(faltan)}" if faltan else ""))
    else:
        print("  --  Seed-VC: falta la carpeta seed-vc/")
    print(f"Idiomas disponibles: {', '.join(IDIOMAS_SOPORTADOS)}")
    return 0


def cmd_extraer(args) -> int:
    from src.audio_extract import guardar_fichas, herramientas_ok, procesar_origen

    print("Herramientas:", herramientas_ok())
    origenes = [args.origen] if args.origen else cfg.VIDEOS
    if not origenes:
        print("Nada que hacer: pasa un origen o rellena VIDEOS en el .env.")
        return 0

    fichas: list[dict] = []
    fallos: list[str] = []
    for origen in origenes:
        try:
            fichas.append(procesar_origen(origen, altura_max=args.altura))
        except Exception as e:
            fallos.append(f"{origen}: {e}")
            print(f"  ERROR  {origen}: {e}")

    for info in fichas:
        print(f"  OK  {info['id']} · {info.get('titulo', info['id'])}")

    # Fusiona con el índice previo (como datos/videos.json de la guía 01 §5)
    previas: dict[str, dict] = {}
    if cfg.FICHA_VIDEOS.exists():
        try:
            contenido = json.loads(cfg.FICHA_VIDEOS.read_text(encoding="utf-8"))
            previas = {f["id"]: f for f in contenido if "id" in f}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Aviso: no se pudo leer {cfg.FICHA_VIDEOS} ({e}); se reescribe.")
    previas.update({f["id"]: f for f in fichas})
    if previas:
        guardar_fichas(list(previas.values()))
        total_min = sum((f.get("duracion_s") or 0) for f in previas.values()) / 60
        print(f"\nOK. Índice en {cfg.FICHA_VIDEOS} · total: {total_min:.1f} minutos")
    if fallos:
        print(f"\n{len(fallos)} origen(es) fallaron.", file=sys.stderr)
        return 1
    return 0


def cmd_separar(args) -> int:
    from src.separate import cargar_pistas, separar

    cancion = _exigir_archivo(args.cancion, "la cancion")
    out = Path(args.salida) if args.salida else cfg.DIR_SEPARACIONES / cancion.stem
    rutas = separar(cancion, out)
    voz, inst, sr = cargar_pistas(rutas)
    print(f"Voz: {rutas['voz']} ({len(voz) / sr:.1f} s)")
    print(f"Instrumental: {rutas['instrumental']} (forma {inst.shape})")
    return 0


def cmd_clonar(args) -> int:
    from src.clone_tts import ESTILOS, IDIOMAS_SOPORTADOS, leer, leer_guion

    voz = _exigir_archivo(args.voz, "la voz de referencia")
    if args.idioma not in IDIOMAS_SOPORTADOS:
        raise ValueError(f"Idioma no soportado: {args.idioma}. "
                         f"Usa: {', '.join(IDIOMAS_SOPORTADOS)}")
    destino = Path(args.salida) if args.salida else cfg.DIR_CLONADOS / "clon.wav"
    if args.guion:
        guion_path = _exigir_archivo(args.guion, "el guion JSON")
        try:
            crudo = json.loads(guion_path.read_text(encoding="utf-8"))
            guion = [(str(f), str(e)) for f, e in crudo]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(
                f"El guion debe ser JSON [[frase, estilo], ...]: {e}") from e
        audio, sr = leer_guion(guion, voz, pausa=args.pausa,
                               guardar=destino, idioma=args.idioma,
                               semilla=args.semilla)
    else:
        if args.estilo not in ESTILOS:
            raise ValueError(f"Estilo no soportado: {args.estilo}")
        audio, sr = leer(args.texto, voz, estilo=args.estilo,
                         pausa=args.pausa, guardar=destino,
                         idioma=args.idioma, semilla=args.semilla)
    print(f"OK: {destino} ({len(audio) / sr:.1f} s a {sr} Hz)")
    return 0


def cmd_convertir(args) -> int:
    from src.sing_convert import (FS, convertir, diferencia_registro,
                                  transposicion_recomendada)
    from src.utils import leer_wav

    fuente = _exigir_archivo(args.fuente, "la voz original")
    ref = _exigir_archivo(args.ref, "tu voz de referencia")
    semitonos = args.semitonos
    if semitonos is None:
        try:
            voz_cancion, sr_c = leer_wav(fuente)
            mi_voz, sr_m = leer_wav(ref)
            dif = diferencia_registro(voz_cancion, mi_voz, sr_c, sr_m)
            semitonos = transposicion_recomendada(float(dif))
            print(f"Diferencia de registro: {dif:+.0f} semitonos -> transponer {semitonos}")
        except ValueError as e:
            semitonos = 0
            print(f"Aviso: no se pudo medir el registro ({e}); se usa 0.", file=sys.stderr)
    destino = Path(args.salida) if args.salida else cfg.DIR_CLONADOS / "voz_convertida.wav"
    audio = convertir(fuente, ref, semitonos=semitonos,
                      pasos=args.pasos, guardar=destino)
    print(f"OK: {destino} ({len(audio) / FS:.1f} s)")
    return 0


def cmd_mezclar(args) -> int:
    from src.mix import cargar_cancion_estereo, mezclar
    from src.utils import leer_wav

    voz_path = _exigir_archivo(args.voz, "la voz")
    base_path = _exigir_archivo(args.base, "la base musical")
    voz, sr_v = leer_wav(voz_path)
    base, sr = cargar_cancion_estereo(base_path)
    if sr_v != sr:
        print(f"Voz a {sr_v} Hz -> se remuestrea a {sr} Hz para mezclar.")
    destino = Path(args.salida) if args.salida else cfg.DIR_MEZCLAS / "mezcla_final.wav"
    mezcla = mezclar(voz, base, sr=sr, sr_voz=sr_v,
                     volumen_voz=args.volumen, guardar=destino)
    print(f"OK: {destino} ({mezcla.shape[1] / sr:.1f} s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    from src.clone_tts import ESTILOS, IDIOMAS_SOPORTADOS

    p = argparse.ArgumentParser(description="Bot clonador de voz")
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    p.add_argument("--debug", action="store_true",
                   help="Muestra la traza completa si algo falla.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("estado", help="Comprueba herramientas y modelos")

    e = sub.add_parser("extraer", help="Descargar video y extraer WAV")
    e.add_argument("origen", nargs="?", default=None,
                   help="URL de YouTube o ruta local. Sin valor: usa VIDEOS del .env")
    e.add_argument("--altura", type=int, default=cfg.ALTURA_MAX)

    s = sub.add_parser("separar", help="Separar voz/instrumental con Demucs")
    s.add_argument("cancion", help="WAV de la cancion (44.1 kHz)")
    s.add_argument("--salida", default=None)

    c = sub.add_parser("clonar", help="Texto -> audio con tu voz (Chatterbox)")
    c.add_argument("--voz", default=str(cfg.MI_VOZ))
    c.add_argument("--texto", default="Hola. Esta frase la genera un modelo con mi voz.")
    c.add_argument("--guion", default=None, help="JSON [[frase, estilo], ...]")
    c.add_argument("--estilo", default="expresivo", choices=list(ESTILOS))
    c.add_argument("--idioma", default="es", choices=IDIOMAS_SOPORTADOS)
    c.add_argument("--semilla", type=int, default=7)
    c.add_argument("--pausa", type=float, default=0.35)
    c.add_argument("--salida", default=None)

    v = sub.add_parser("convertir", help="Voz cantada -> tu timbre (Seed-VC)")
    v.add_argument("--fuente", required=True, help="Voz original separada")
    v.add_argument("--ref", required=True, help="Tu voz (mi_voz.wav)")
    v.add_argument("--semitonos", type=int, default=None,
                   help="0, -12 o 12. Sin valor: se recomienda solo.")
    v.add_argument("--pasos", type=int, default=None)
    v.add_argument("--salida", default=None)

    m = sub.add_parser("mezclar", help="Voz + base musical")
    m.add_argument("--voz", required=True)
    m.add_argument("--base", required=True)
    m.add_argument("--volumen", type=float, default=1.0)
    m.add_argument("--salida", default=None)

    args = p.parse_args(argv)
    comandos = {"estado": cmd_estado, "extraer": cmd_extraer,
                "separar": cmd_separar, "clonar": cmd_clonar,
                "convertir": cmd_convertir, "mezclar": cmd_mezclar}
    try:
        return int(comandos[args.cmd](args))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError,
            KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            raise
        return 1
    except KeyboardInterrupt:
        print("Interrumpido por el usuario.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
