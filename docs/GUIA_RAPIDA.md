# Guia rapida

## 1. Extraer audio de un video

- Pega la URL de YouTube (solo videos tuyos, CC o con permiso) o sube un mp4/mp3.
- Sale un WAV mono 16 kHz en `data/audios/`, que es lo que esperan los modelos.
- Comprueba el dibujo de la onda y escucha los primeros segundos.

## 2. Registrar tu voz

- Lee el texto de ejemplo en voz natural, 10-20 segundos, sin eco.
- La app revisa 3 cosas: duracion >= 8 s, volumen audible, sin saturacion.
- Queda en `data/referencias/mi_voz.wav` y se reutiliza en los pasos 3 y 4.

## 3. Clonar voz hablada

- Escribe el texto **para ser oido**: frases cortas, comas, `¿?` y `¡!`.
- Estilos: `plano` (0.25) / `natural` (0.5) / `expresivo` (0.7) / `teatral` (0.9).
- Si una frase sale rara, cambia la semilla antes de tocar los mandos.
- Textos largos: frase a frase con su estilo y pausa de 0.35 s.

## 4. Poner tu voz a cantar

1. Sube la cancion base (con voz + acompanamiento).
2. Demucs la parte en `voz` e `instrumental` (revisa con auriculares:
   restos de eco ensucian la conversion).
3. Seed-VC convierte la voz a tu timbre. Regla de tono:
   - Diferencia <= 8 semitonos: dejar en 0.
   - Cancion mucho mas aguda que tu voz: transponer `-12` (una octava abajo,
     sigue encajando con el instrumental).
4. Compara original vs convertida antes de mezclar.

## 5. Mezcla final

- Volumen 1.0 = nivel original; 1.3 sube la voz, 0.8 la baja.
- La mezcla iguala niveles, recorta duracion y evita saturacion (pico 0.98).

## Si algo falla

| Sintoma | Causa probable | Que hacer |
|---|---|---|
| Suena metalico | Restos de musica en la pista de voz | Fragmento con voz mas sola |
| Dos voces / borroso | Coros en el original | Seed-VC convierte una voz cada vez |
| Falsete forzado | Melodia fuera de tu registro | Transponer -12 o +12 |
| No se parece a ti | Referencia corta o con ruido | Regrabar con mas entonacion |
| Error 403 YouTube | Bloqueo temporal | `pip install -U yt-dlp`, otra red, o archivo local |
