# Clonador de voz

Extrae el audio de un video, clona tu voz y ponla sobre una base musical.
Bot Python + app Streamlit, con tres modelos de codigo abierto.

| Paso | Que hace | Modelo |
|---|---|---|
| 1. Extraer | Video de YouTube/archivo a WAV mono 16 kHz | `yt-dlp` + `ffmpeg` |
| 2. Mi voz | Referencia de 10-20 s validada (duracion/volumen/pico) | — |
| 3. Clonar | Texto a audio con tu voz, 23 idiomas | Chatterbox Multilingual |
| 4. Cantar | Voz de una cancion con tu timbre | Demucs + Seed-VC |
| 5. Mezclar | Voz clonada + base musical de fondo | — |

## Configuración (.env)

Como pide la guía (cuaderno 01), el personaje y los vídeos se cambian en
un solo lugar: el archivo `.env` de la raíz (no se sube a git).

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Edita `PERSONAJE`, `DESCRIPCION` y `VIDEOS` (URLs o rutas, separadas por
comas). El bot y la app lo leen solos:

```bash
python scripts/bot.py extraer   # procesa todos los VIDEOS del .env
```

## Estructura

```
app.py                 # App Streamlit (punto de entrada)
scripts/bot.py         # Bot CLI (mismo flujo sin interfaz)
src/
  config.py            # Rutas centrales data/ + frecuencias
  audio_extract.py     # Paso 1: yt-dlp + ffmpeg
  utils.py             # WAV, validacion, normalizacion (solo numpy+soundfile)
  clone_tts.py         # Paso 3: Chatterbox (carga perezosa)
  separate.py          # Paso 4a: Demucs voz/instrumental
  sing_convert.py      # Paso 4b: Seed-VC conversion cantada
  mix.py               # Paso 5: mezcla voz + base
data/                  # videos, audios, referencias, separaciones, clonados, mezclas
Archivos/              # Cuadernos y guias originales del tutorial
docs/                  # Guia rapida de uso
tests/                 # Tests unitarios
```

## Uso rapido

```bash
pip install -r requirements.txt
streamlit run app.py
```

Bot CLI (sin interfaz):

```bash
python scripts/bot.py extraer URL_O_ARCHIVO
python scripts/bot.py clonar --voz data/referencias/mi_voz.wav --texto "Hola, mundo"
python scripts/bot.py separar data/audios/mi_audio.wav
python scripts/bot.py convertir --fuente data/separaciones/voz.wav --ref data/referencias/mi_voz.wav
python scripts/bot.py mezclar --voz data/clonados/voz.wav --base data/separaciones/inst.wav
```

## Entornos con modelos

Chatterbox y Seed-VC piden versiones distintas: van en entornos separados.

```bash
# Voz hablada (Python 3.11)
pip install -r requirements-voz.txt

# Voz cantada (Python 3.10) + carpeta seed-vc junto a la raiz:
git clone https://github.com/Plachtaa/seed-vc.git
pip install -r requirements-canto.txt
```

La app detecta que falta y te dice como instalarlo; los demas pasos
(extraer, validar, mezclar) funcionan sin GPU ni modelos.

## Requisitos externos

- `ffmpeg` (+ `ffprobe`): Windows `winget install ffmpeg`, macOS `brew install ffmpeg`.
- GPU opcional: en CPU la conversion cantada tarda ~10 min por 30 s;
  usa fragmentos de 10-15 s.

## Uso responsable

Clona solo tu voz o la de quien te haya dado permiso, usa canciones
libres o tuyas, y avisa siempre de que el audio es generado.
Chatterbox anade marca de agua detectable; Seed-VC no anade ninguna.
