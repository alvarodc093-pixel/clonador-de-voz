# Clonador de voz

Extrae el audio de un video, clona tu voz y ponla sobre una base musical.
Bot Python + app Streamlit, con modelos de codigo abierto.

| Paso | Que hace | Modelo |
|---|---|---|
| 1. Extraer | Video de YouTube/archivo a WAV mono 16 kHz | `yt-dlp` + `ffmpeg` |
| 2. Mi voz | Referencia de 10-20 s validada (duracion/volumen/pico) | — |
| 3. Clonar | Texto a audio con tu voz, 23 idiomas | Chatterbox Multilingual |
| 4. Cantar | Voz de una cancion con tu timbre | Demucs + Seed-VC |
| 5. Mezclar | Voz clonada + base musical de fondo | — |

Los pasos 1, 2 y 5 funcionan sin GPU ni modelos. Los pasos 3 y 4
descargan modelos la primera vez que se usan.

## Configuracion (.env)

El personaje y los videos se cambian en un solo lugar: el archivo `.env`
de la raiz (no se sube a git).

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Edita `PERSONAJE`, `DESCRIPCION` y `VIDEOS` (URLs o rutas, separadas por
comas). El bot y la app lo leen solos:

```bash
python scripts/bot.py extraer   # procesa todos los VIDEOS del .env
```

Variables reconocidas: `PERSONAJE`, `DESCRIPCION`, `VIDEOS`, `ALTURA_MAX`,
`FS_VOZ` (16000) y `FS_MUSICA` (44100).

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Requisitos externos:

- `ffmpeg` + `ffprobe`: Windows `winget install ffmpeg`,
  macOS `brew install ffmpeg`, Linux `sudo apt install ffmpeg`.
  La app tambien los busca en la carpeta de WinGet si el `PATH` no se
  actualizo.

## Uso rapido

```bash
streamlit run app.py
```

Bot CLI (sin interfaz):

```bash
python scripts/bot.py estado                          # que hay instalado
python scripts/bot.py extraer URL_O_ARCHIVO           # o sin args: VIDEOS del .env
python scripts/bot.py clonar --voz data/referencias/mi_voz.wav --texto "Hola, mundo"
python scripts/bot.py separar data/audios/mi_audio.wav
python scripts/bot.py convertir --fuente data/separaciones/voz.wav --ref data/referencias/mi_voz.wav
python scripts/bot.py mezclar --voz data/clonados/voz.wav --base data/separaciones/inst.wav
```

`python scripts/bot.py <comando> --help` muestra todas las opciones.

## Entornos con modelos

Chatterbox y Seed-VC piden versiones distintas: van en entornos separados.
La app arranca sin ellos y explica como instalarlos; el panel lateral
muestra que falta en cada caso.

```bash
# Voz hablada (Python 3.11)
pip install -r requirements-voz.txt

# Voz cantada (Python 3.10) + carpeta seed-vc junto a la raiz:
git clone https://github.com/Plachtaa/seed-vc.git
pip install -r requirements-canto.txt
```

Para usar Seed-VC **desde la app**, ejecuta la app en el mismo entorno
donde instalaste `requirements-canto.txt` (Streamlit ya viene incluido en
ese archivo). El panel lateral marca `Seed-VC` como no disponible mientras
falten paquetes de canto (`munch`, `descript-audio-codec`, etc.).

## Estructura

```
app.py                 # App Streamlit (punto de entrada)
scripts/bot.py         # Bot CLI (mismo flujo sin interfaz)
src/
  config.py            # Rutas centrales data/ + frecuencias + .env
  audio_extract.py     # Paso 1: yt-dlp + ffmpeg (con fallback sin ffmpeg)
  utils.py             # WAV, validacion, normalizacion, remuestreo
  clone_tts.py         # Paso 3: Chatterbox (carga perezosa)
  separate.py          # Paso 4a: Demucs voz/instrumental
  sing_convert.py      # Paso 4b: Seed-VC conversion cantada
  mix.py               # Paso 5: mezcla voz + base
data/                  # videos, audios, referencias, separaciones, clonados, mezclas
Archivos/              # Cuadernos y guias originales del tutorial
docs/                  # Guia rapida de uso
tests/                 # Tests unitarios + AppTest de Streamlit
```

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest -q
```

Los tests cubren utilidades de audio, mezcla, extraccion local, separacion,
clonacion (con modelo simulado), conversion (con motor simulado) y el
arranque de la app con `streamlit.testing.v1.AppTest`. No descargan
modelos reales.

## Requisitos y limitaciones

- GPU opcional: en CPU la conversion cantada tarda ~10 min por 30 s.
  Por eso el paso 4 permite convertir solo los primeros 30/60 s.
- Seed-VC recorta la referencia a 25 s y procesa el audio por trozos.
- Demucs y Seed-VC descargan pesos la primera vez (varios cientos de MB).
- La mezcla recorta a la pista mas corta e iguala el pico a 0.98.
- La referencia de voz se normaliza de pico para que el modelo reciba un
  nivel constante.

## Uso responsable

Clona solo tu voz o la de quien te haya dado permiso, usa canciones
libres o tuyas, y avisa siempre de que el audio es generado.
Chatterbox anade marca de agua detectable; Seed-VC no anade ninguna.
