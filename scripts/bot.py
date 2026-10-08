"""Bot CLI del clonador de voz.

Une los 4 pasos de las guias en comandos sueltos (cada uno deja sus
archivos en data/ para el siguiente):

    python scripts/bot.py extraer URL_O_ARCHIVO   # o sin args: usa VIDEOS del .env
    python scripts/bot.py separar data/audios/01_fragmento.wav
    python scripts/bot.py clonar --voz data/referencias/mi_voz.wav --texto "Hola, mundo"
    python scripts/bot.py convertir --fuente data/separaciones/voz.wav --ref data/referencias/mi_voz.wav
    python scripts/bot.py mezclar --voz data/clonados/voz.wav --base data/separaciones/inst.wav

La app Streamlit (app.py) llama a las mismas funciones de src/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402


def cmd_extraer(args) -> None:
    import json

    from src.audio_extract import guardar_fichas, herramientas_ok, procesar_origen

    print("Herramientas:", herramientas_ok())
    origenes = [args.origen] if args.origen else cfg.VIDEOS
    if not origenes:
        print("Nada que hacer: pasa un origen o rellena VIDEOS en el .env.")
        return
    fichas = [procesar_origen(o, altura_max=args.altura) for o in origenes]
    for info in fichas:
        print(f"  ✅ {info['id']} · {info.get('titulo', info['id'])}")
    # Fusiona con el índice previo (como datos/videos.json de la guía 01 §5)
    previas = {}
    if cfg.FICHA_VIDEOS.exists():
        previas = {f["id"]: f for f in json.loads(cfg.FICHA_VIDEOS.read_text(encoding="utf-8"))}
    previas.update({f["id"]: f for f in fichas})
    guardar_fichas(list(previas.values()))
    total_min = sum((f.get("duracion_s") or 0) for f in previas.values()) / 60
    print(f"\nOK. Índice en {cfg.FICHA_VIDEOS} · total: {total_min:.1f} minutos")


def cmd_separar(args) -> None:
    from src.separate import cargar_pistas, separar

    out = Path(args.salida) if args.salida else cfg.DIR_SEPARACIONES / Path(args.cancion).stem
    rutas = separar(args.cancion, out)
    voz, inst, sr = cargar_pistas(rutas)
    print(f"Voz: {rutas['voz']} ({len(voz) / sr:.1f} s)")
    print(f"Instrumental: {rutas['instrumental']} (forma {inst.shape})")


def cmd_clonar(args) -> None:
    from src.clone_tts import leer, leer_guion

    destino = Path(args.salida) if args.salida else cfg.DIR_CLONADOS / "clon.wav"
    if args.guion:
        import json

        guion = [(f, e) for f, e in json.loads(Path(args.guion).read_text(encoding="utf-8"))]
        audio, sr = leer_guion(guion, args.voz, pausa=args.pausa,
                               guardar=destino, idioma=args.idioma)
    else:
        audio, sr = leer(args.texto, args.voz, estilo=args.estilo,
                         pausa=args.pausa, guardar=destino, idioma=args.idioma)
    print(f"OK: {destino} ({len(audio) / sr:.1f} s a {sr} Hz)")


def cmd_convertir(args) -> None:
    from src.sing_convert import convertir, diferencia_registro, transposicion_recomendada
    from src.utils import leer_wav

    semitonos = args.semitonos
    if semitonos is None:
        voz_cancion, sr_c = leer_wav(args.fuente)
        mi_voz, sr_m = leer_wav(args.ref)
        dif = diferencia_registro(voz_cancion, mi_voz, sr_c, sr_m)
        semitonos = transposicion_recomendada(float(dif))
        print(f"Diferencia de registro: {dif:+.0f} semitonos -> transponer {semitonos}")
    destino = Path(args.salida) if args.salida else cfg.DIR_CLONADOS / "voz_convertida.wav"
    audio = convertir(args.fuente, args.ref, semitonos=semitonos,
                      pasos=args.pasos, guardar=destino)
    print(f"OK: {destino} ({len(audio) / 44100:.1f} s)")


def cmd_mezclar(args) -> None:
    from src.mix import cargar_cancion_estereo, mezclar
    from src.utils import leer_wav

    voz, sr_v = leer_wav(args.voz)
    base, sr = cargar_cancion_estereo(args.base)
    if sr_v != sr:
        print(f"Voz a {sr_v} Hz -> se remuestrea a {sr} Hz para mezclar.")
    destino = Path(args.salida) if args.salida else cfg.DIR_MEZCLAS / "mezcla_final.wav"
    mezcla = mezclar(voz, base, sr=sr, sr_voz=sr_v,
                     volumen_voz=args.volumen, guardar=destino)
    print(f"OK: {destino} ({mezcla.shape[1] / sr:.1f} s)")


def main() -> None:
    p = argparse.ArgumentParser(description="Bot clonador de voz")
    sub = p.add_subparsers(dest="cmd", required=True)

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
    c.add_argument("--estilo", default="expresivo",
                   choices=["plano", "natural", "expresivo", "teatral"])
    c.add_argument("--idioma", default="es")
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

    args = p.parse_args()
    {"extraer": cmd_extraer, "separar": cmd_separar, "clonar": cmd_clonar,
     "convertir": cmd_convertir, "mezclar": cmd_mezclar}[args.cmd](args)


if __name__ == "__main__":
    main()
