# Visión en planta: demo de analítica de video con IA

La demo toma el video de una cámara y muestra en el navegador:

- **Quién está en cada zona y cuánto tiempo.** Las zonas se dibujan sobre la imagen: estaciones de trabajo y zonas restringidas.
- **Métricas por zona:** tiempo ocupada y vacía, % de ocupación, visitas, permanencia promedio y máxima, línea de tiempo y cantidad de personas.
- **Alertas:** permanencia excesiva, estación vacía demasiado tiempo e ingreso a una zona restringida.
- **Ergonomía y caídas:** brazos sobre los hombros, tronco flexionado y persona en el piso.
- **Mapa de calor** de dónde estuvo la gente.
- **Identificación de personas registradas con su consentimiento**, opcional: viene apagada y se prende y apaga desde la web. Apagada, todo funciona en forma anónima y no se procesa ninguna cara.

Todo queda guardado (SQLite), así que las métricas se pueden consultar por rango horario aunque se reinicie el sistema.
Corre en Docker; lo único que corre en Windows es el puente que publica la webcam (ver más abajo por qué).

---

## Requisitos

| | |
|---|---|
| Sistema | Windows 10/11 con **Docker Desktop** (backend WSL2) |
| GPU (recomendado) | NVIDIA con driver actualizado. Probado con una RTX 3060 Ti 8 GB. Sin GPU, ver [Modo sin GPU](#modo-sin-gpu) |
| Disco | ~10 GB para la imagen con GPU (2,4 GB la de CPU) |
| Cámara | La webcam de la PC o cualquier cámara IP con RTSP |

La primera vez la imagen se construye sola: baja PyTorch con CUDA y los modelos, y tarda varios minutos. Después el
contenedor arranca sin internet, porque los modelos quedan dentro de la imagen.

## Puesta en marcha

```powershell
# 1. Levantar el servidor de video y la app (la primera vez construye la imagen)
docker compose up -d --build

# 2. En otra terminal: publicar la webcam (dejarla abierta; Ctrl+C para cortar)
.\webcam.bat

# 3. Abrir http://localhost:8000
```

`webcam.bat` la primera vez baja **ffmpeg portable** a `tools\` (~115 MB, con checksum verificado; no instala nada ni
toca el PATH). Opciones útiles:

```powershell
.\webcam.bat -List                               # cámaras disponibles
.\webcam.bat -Device "UVC Camera" -Size 640x480  # elegir cámara y resolución (por defecto 1280x720 a 30 fps)
```

Si no aparece el video:
- Windows → Configuración → Privacidad → Cámara: habilitar *"Permitir que las aplicaciones de escritorio accedan a la cámara"*.
- Una webcam la puede usar **una sola aplicación a la vez**: cerrar Teams, Zoom, OBS, etc.
- El orden no importa: si la app arranca antes que la webcam, reconecta sola en unos segundos (y lo mismo si se corta el puente).

**¿Por qué la webcam no va en Docker?** Docker Desktop en Windows no puede acceder a dispositivos USB. El puente
convierte la webcam en una "cámara IP" (RTSP), así que la app la trata igual que a una cámara de planta: pasar a
cámaras reales es cambiar una URL.

### Comandos del día a día

```powershell
docker compose logs -f app      # ver qué está haciendo (Ctrl+C sale del log, no apaga nada)
docker compose restart app      # aplicar cambios de config.yaml
docker compose down             # apagar (los datos quedan)
docker compose up -d            # volver a prender
```

---

## Uso

### Pestaña Zonas: marcar las estaciones

1. **Nueva zona** y click en cada vértice sobre la imagen. Para cerrar: doble click, Enter o click en el primer
   vértice. Retroceso o Z deshace el último punto y Esc cancela.
2. Arrastrar los vértices para ajustar. Poner un nombre (por ejemplo, *Lijado*) y elegir el tipo:
   - **Estación de trabajo:** con dos alertas opcionales, *"si alguien se queda más de N s"* y *"si queda vacía más de N s"*. Si un campo queda vacío, esa alerta no se usa.
   - **Zona restringida:** cualquier ingreso genera una alerta.
3. **Guardar.** Se aplica al instante, sin reiniciar.

**Exportar** e **Importar** guardan y cargan las zonas como JSON. Importar *reemplaza* todas las zonas de la cámara.

Una persona está "dentro" de una zona según **un punto de su cuerpo**, que se configura en `camera.anchor`:
- `torso` es el que conviene para la webcam de escritorio, porque no se ven los pies.
- `feet` es el más preciso para una cámara en altura que mira al piso.

### Pestaña En vivo

Video con las zonas, cada persona con su ID (o su nombre) y su tiempo en la zona, alertas activas y estado del sistema.
Una zona con alerta se pinta de rojo. Para no contar de más o de menos:
- una entrada cuenta recién después de 0,5 s dentro;
- una salida se registra después de 3 s sin ver a la persona, así un parpadeo del detector o alguien que pasa por
  delante no corta la visita.

### Pestaña Métricas

- **Rango:** últimos 15 min, última hora, últimas 8 h, hoy, o un rango personalizado. Se actualiza cada 5 s.
- **Ocupación por zona:** ocupada, vacía, % de ocupación, visitas, permanencia promedio y máxima, tiempo-persona y alertas.
- **Línea de tiempo de ocupación** y **personas en zonas**.
- **Mapa de calor de permanencia** sobre la imagen actual de la cámara.
- **Por persona**, solo si hay personas identificadas.
- **Alertas del período.**

Cada gráfico tiene **Ver como tabla**. Los períodos en que el sistema estuvo apagado o sin video se muestran como
*sin video* y no cuentan como "vacía".

### Pestaña Personas: identificación con consentimiento

1. Escribir el nombre y tildar el **consentimiento**: no se puede registrar a nadie sin tildarlo.
2. La persona se para frente a la cámara, sola y a menos de 1,5 m, y se toca **Capturar y registrar**. El sistema toma 5
   muestras en unos segundos. Si algo falla, dice qué: la cara se ve chica, hay más de una cara, no hay buena luz.
3. Con el interruptor **Identificación de personas** prendido, el nombre aparece sobre la persona en el video y en las
   métricas. Mientras está prendido, la barra superior y el video lo avisan en rojo.

Cómo está pensada:
- **No se guardan fotos**, solo un vector numérico de la cara.
- **Borrar a una persona** elimina ese vector y deja su historial anónimo: las visitas siguen contando, pero sin nombre.
- Si la cara no se ve bien, la persona queda anónima. El sistema nunca adivina un nombre (ver [Qué se midió](#qué-se-midió)).

### Ergonomía y caídas

Usa un segundo modelo que estima 17 puntos del cuerpo. Genera tres alertas, cada una cuando la condición se sostiene un tiempo:

| Alerta | Condición (configurable en `posture`) |
|---|---|
| Brazos sobre los hombros | una muñeca por encima del hombro más de 5 s |
| Tronco flexionado | torso inclinado ≥ 45° más de 5 s |
| Posible caída | persona acostada (cuerpo horizontal y caja más ancha que alta) más de 2 s |

Necesita **ver el cuerpo entero**: con la webcam de escritorio, a unos 2-3 m. Mide sobre la imagen, así que depende del
ángulo de la cámara. Es un **aviso de apoyo, no un sistema de seguridad** (ver la tasa medida más abajo).

---

## Configuración

`config.yaml` se monta en solo lectura en el contenedor. Después de editarlo hay que correr `docker compose restart app`.
Una clave mal escrita hace que la app no arranque y el log dice cuál es: no se ignora en silencio.

| Sección | Para qué |
|---|---|
| `camera` | `id` de la cámara, `source` (URL del video) y `anchor` (`torso` / `feet` / `center`) |
| `detector` | modelo (`auto` = `yolo26s` con GPU, `yolo26n` con CPU), confianza y memoria del tracker |
| `occupancy` | tiempo para contar una entrada (0,5 s) y gracia antes de dar la salida (3 s) |
| `identification` | cómo arranca (`enabled: false`), umbral de reconocimiento (0,45) y margen sobre el segundo candidato (0,08) |
| `posture` | prender o apagar la ergonomía y los umbrales de cada alerta |
| `stream` | ancho, fps y calidad del video que se manda al navegador |

### Cambiar la fuente de video

Sin editar el archivo, con una variable de entorno:

```powershell
$env:CAMERA_SOURCE = "rtsp://mediamtx:8554/demo"; docker compose up -d     # video de prueba (gente caminando)
$env:CAMERA_SOURCE = ""; docker compose up -d                              # volver a lo que diga config.yaml
```

El video de prueba no viene en el repositorio, porque es un ejemplo de OpenCV. Se baja una vez:

```powershell
New-Item -ItemType Directory -Force media | Out-Null
Invoke-WebRequest https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi -OutFile media\vtest.avi
```

**Cámara IP de planta:** en `config.yaml`, `camera.source: rtsp://usuario:clave@192.168.1.50:554/stream1`.
- La ruta exacta depende de la marca y suele estar en el manual. Conviene el *substream*, de menor resolución, salvo que se use identificación.
- Si la clave tiene caracteres especiales, hay que codificarlos: `@` → `%40`, `:` → `%3A`, `/` → `%2F`.
- La clave nunca se muestra en la web ni en los logs.
- **Si cambiás de cámara, cambiá también `camera.id`.** Las zonas y las métricas se guardan por cámara: con el mismo
  id, las zonas dibujadas para la webcam aparecerían sobre la imagen de la otra cámara.

**Video grabado** (por ejemplo, un minuto de una cámara de planta para ver cómo se comporta): copiarlo a `media\` y
usar `$env:CAMERA_SOURCE = "/media/archivo.mp4"`. Se reproduce en loop a su velocidad real.

### Abrir la web desde otra PC

Por defecto solo responde en esta PC, porque el video muestra personas. Para abrirla en la red:

```powershell
$env:BIND_ADDR = "0.0.0.0"; docker compose up -d     # después: http://<IP de esta PC>:8000
```

La demo **no tiene usuarios ni contraseña**: cualquiera en la red que conozca la dirección ve el video.

### Modo sin GPU

```powershell
docker compose -f compose.yaml -f compose.cpu.yaml up -d --build
```

Usa PyTorch sin CUDA, que da una imagen de 2,4 GB, y los modelos livianos. En un Ryzen 5 5600X procesa unos 8-9 cuadros por
segundo con la ergonomía prendida. Alcanza para la demo; en una PC más lenta conviene `posture.enabled: false`.

---

## Datos

Todo está en una base SQLite dentro del volumen de Docker `vision-data`. No está en una carpeta de Windows porque
SQLite en carpetas compartidas con Windows tiene problemas de bloqueo. Se guarda:
- zonas;
- visitas con entrada y salida;
- alertas;
- sesiones: cuándo el sistema estuvo mirando;
- personas registradas y sus vectores faciales;
- posiciones para el mapa de calor, que se borran solas a los 30 días.

**Backup, con la app andando:**

```powershell
docker compose exec app python -c "import sqlite3; sqlite3.connect('/data/vision.db').backup(sqlite3.connect('/data/backup.db'))"
docker compose cp app:/data/backup.db .\vision-backup.db
```

**Restaurar un backup:**

```powershell
docker compose stop app
docker compose run --rm --no-deps -v "${PWD}:/backup" app sh -c "cp /backup/vision-backup.db /data/vision.db && rm -f /data/vision.db-wal /data/vision.db-shm"
docker compose start app
```

Si el backup se tomó con la app andando, al arrancar el log avisa que "no se cerró bien la vez anterior". Es normal:
cierra la sesión que estaba abierta en el momento del backup.

**Borrar todo y empezar de cero:** `docker compose down -v`. Elimina el volumen con todos los datos y no se puede deshacer.

Si la app se cortó de golpe (corte de luz, `docker kill`), al arrancar cierra las visitas, alertas y sesiones que
quedaron abiertas y lo anota en el log.

## Tests

```powershell
docker compose run --rm app pytest -q
```

Son 158 tests y corren dentro del contenedor, sin instalar nada en Windows. La lógica de negocio está separada del video y de
los modelos, así que se prueba con casos armados. Esa lógica incluye ocupación, alertas, métricas, votación de
identidad y posturas. Los casos cubren parpadeos, cambios de ID del tracker, zonas editadas en caliente, cortes de la
fuente, rangos que cortan visitas abiertas y otros.

## Herramientas de medición (`scripts/`)

Los scripts usan el mismo detector, tracker, motor facial y reglas que la app. Corren dentro del contenedor y sirven para medir antes de comprometerse:

| Script | Qué responde |
|---|---|
| `viabilidad_facial.py` | **¿Esta cámara sirve para reconocer caras?** Se graba ~1 min con la cámara candidata, con gente moviéndose como en el día a día, y el script mide el tamaño de las caras, cuántas llegan al mínimo y si las huellas distinguen personas. Termina con un veredicto. |
| `evaluar_reconocimiento.py` | Precisión del reconocimiento con LFW (fotos públicas de miles de personas): para recalibrar el umbral. |
| `evaluar_caidas.py` | Caídas detectadas y falsas alarmas con el UR Fall Detection Dataset. |

```powershell
# con el video copiado en media\
Get-Content scripts\viabilidad_facial.py -Raw | docker compose exec -T app python - /media/video.mp4
```

El docstring de cada script explica cómo bajar su dataset. Los datasets no se suben al repositorio.

---

## Qué se midió

Mediciones hechas en esta PC: Ryzen 5 5600X y RTX 3060 Ti.

**Velocidad**

| | GPU (RTX 3060 Ti) | CPU (Ryzen 5 5600X) |
|---|---|---|
| Detección y seguimiento de personas, 720p | ~18 ms (`yolo26s`) | ~45-65 ms (`yolo26n`) |
| Pose para ergonomía, cada 2 cuadros | ~18 ms (`yolo26s-pose`) | ~55-70 ms (`yolo26n-pose`) |
| Cuadros analizados por segundo, en vivo y con ergonomía | 23-26 (webcam a 30 fps) | 7,6-9 (video de prueba de 10 fps) |

**Identificación facial**
- Con LFW, 25 personas registradas, 465 fotos de esas personas y 300 desconocidos, con el umbral 0,45:
  - reconoce el 99,8 % de las fotos;
  - no confunde a ninguna persona con otra;
  - acepta como registrado a un desconocido el 0,3 % de las veces.
- De punta a punta, con 70 apariciones simuladas: 45 de 45 personas registradas identificadas, ningún nombre
  equivocado y ningún desconocido con nombre.
- Cuando el tracker intercambia los IDs de dos personas, el nombre equivocado dura ~1,2-1,4 s hasta que la reverificación lo retira.
- **Con una cámara de tipo planta no funciona.** En el video del cruce peatonal (cámara en altura y gente a varios metros):
  - las caras miden 7-10 px de ancho;
  - ninguna de las 4.671 detecciones llega al mínimo;
  - si se forzara el reconocimiento, confundiría el 7,4 % de las veces.

  Por eso la app no intenta reconocer caras chicas: la persona queda anónima.
- Para reconocer hacen falta caras de **60-80 px de ancho y de frente**. Eso da una cámara a la altura de la cara
  que apunta a un punto de paso, como un molinete o una puerta, no una cámara general de nave.

**Caídas** (UR Fall Detection Dataset, 20 caídas y 20 actividades normales)
- Detecta **13 de 20 caídas (65 %)**.
- Da 2 falsas alarmas en 20. Las dos son personas acostadas en un sillón, que en la imagen es igual a estar tirado.
- En el video del cruce, con gente caminando, dio 0 alertas de postura.
- Sirve como aviso adicional, **no reemplaza** un sistema de seguridad. Mejorarlo requiere entrenar con video propio de la planta.

**Limitaciones conocidas**
- Cuando dos personas se cruzan o alguien queda tapado mucho tiempo, el tracker puede darle un ID nuevo. Eso **puede
  inflar la cantidad de visitas**. El tiempo ocupada y vacía no se ve afectado, porque se calcula por unión de intervalos.
- Las posturas se miden en 2D, sobre la imagen: una misma inclinación se ve distinta según dónde esté la cámara.
- Una cámara y un proceso; sin usuarios, sin HTTPS.

---

## Arquitectura de la demo

```
Windows    webcam ─► tools\ffmpeg.exe (H.264 baja latencia) ─► rtsp://127.0.0.1:8554/webcam
Docker     mediamtx   servidor RTSP (webcam publicada, video de prueba en loop)
           app        FastAPI, 1 proceso, GPU
             hilo de captura   último cuadro, reconexión automática
             hilo de análisis  detección + tracking → zonas → ocupación → alertas
                               → identificación (opcional) → pose → overlay
             hilo escritor     SQLite (volumen vision-data)
             web               video MJPEG + estado por WebSocket (4 veces por segundo) + API REST
```

```
app/          main.py (web), pipeline.py (bucle de análisis), capture.py, detector.py, occupancy.py, alerts.py,
              posture.py, identity.py, faces.py, metrics.py, storage.py, recorder.py, overlay.py, config.py
app/static/   interfaz web (HTML + JS sin compilación; gráficos propios en SVG/Canvas)
tests/        pytest
scripts/      webcam.ps1 y herramientas de medición
```

---

## Llevarlo a producción

### ¿Dónde corre? Arquitectura en planta

Las cámaras no se conectan "al servidor": se conectan a la **red**, con un switch PoE que les da datos y alimentación
por el mismo cable. El servidor **lee** el video de cada una por RTSP. Si la planta ya tiene un grabador (NVR), se
puede leer de él sin tocar las cámaras. Cada cámara es una URL en la configuración.

El análisis tiene que hacerse donde está el video, o el video tiene que viajar hasta donde se analiza. Hay tres formas:

| | Servidor en planta **(recomendado)** | Edge (un equipo chico por sector) | Nube |
|---|---|---|---|
| Qué es | Una PC o servidor con GPU NVIDIA en la red de la planta | Equipos como NVIDIA Jetson junto a las cámaras; mandan solo eventos | Las cámaras mandan video a un servidor con GPU en la nube |
| Video | No sale de la planta | No sale del sector | Sale por internet, 24/7 |
| Internet | No hace falta para analizar | No hace falta | Imprescindible: si se corta, no hay monitoreo |
| Costo | Una inversión inicial | Varios equipos chicos | Abono mensual de GPU más ancho de banda de subida |
| Cuándo | Una planta, con cámaras en la misma red | Naves alejadas o red débil | Varias plantas sin personal de sistemas local |

**No es obligatorio un servidor físico dedicado.** Lo que hace falta es una GPU cerca del video. Puede ser:
- una PC con una placa como la de esta demo;
- una máquina virtual con GPU en la virtualización que ya tenga la planta;
- un servidor de rack.

Para un piloto de 1 a 4 cámaras alcanza con esta misma PC. En producción conviene **Linux** con NVIDIA Container
Toolkit: el mismo `compose.yaml` funciona y no hace falta el puente de webcam.

**La nube es posible, pero no conviene para analizar video.**
- **Ancho de banda:** cada cámara manda ~1-2 Mbps en forma continua. Con 20 cámaras son 20-40 Mbps de subida, las 24 horas.
- **Costo:** una GPU en la nube encendida todo el mes cuesta, en pocos meses, lo que una placa propia.
- **Legal:** el video de los empleados sale de la planta; si el proveedor guarda datos fuera del país, se suma la
  transferencia internacional de datos personales.
- **Cómo se conectaría:** nunca abriendo las cámaras a internet, porque son equipos muy atacados. Un equipo en
  la planta tendría que reenviar el video por un túnel saliente: VPN tipo WireGuard o un relay RTSP/SRT.

**Lo que sí tiene sentido en la nube es el resultado.**
- Analizar en planta y mandar a la nube **solo eventos y métricas**: kilobytes, no video.
- Así el dashboard se ve desde cualquier lado y se puede **integrar con Odoo**: tiempos por estación junto a las
  órdenes de producción, alertas como actividades.

**Dimensionamiento.** Es una estimación a partir de lo medido; hay que confirmarla con una prueba de carga.
- Para ocupación y tiempos, 5 cuadros por segundo por cámara alcanzan.
- Con ~18 ms por cuadro, una GPU como la de esta PC atendería del orden de **10 cámaras sin ergonomía o 5 con ergonomía**.
- Procesar en lotes y usar el substream de cada cámara sube ese número.
- Con muchas cámaras, la decodificación del video conviene hacerla en la GPU (NVDEC).

**Red.**
- Cámaras en una **VLAN aislada**, sin salida a internet.
- El servidor accede a esa VLAN y la web se publica en la red de oficina.

### Lo demás que falta

**Software**
- **Varias cámaras.** El modelo de datos ya tiene `camera_id` en zonas, visitas y alertas. Falta correr un análisis por cámara compartiendo la GPU y un selector de cámara en la web.
- **Base de datos.** Pasar a PostgreSQL o TimescaleDB, con backups automáticos.
- **Seguridad.**
  - Usuarios con roles, porque no todos deberían ver el video ni la pestaña Personas.
  - HTTPS.
  - Registro de quién vio qué.
- **Operación.**
  - Monitoreo de que cada cámara esté viva, que hoy se ve en `/api/health`.
  - Alertas por mail, Teams o una actividad en Odoo, no solo en pantalla.
  - Reinicio automático y actualizaciones.

**Licencias**
- El detector es **Ultralytics YOLO (AGPL-3.0)**. Para la demo no hay problema. Para uso productivo hay dos caminos:
  - la licencia **Enterprise** de Ultralytics;
  - cumplir la AGPL, que entre otras cosas obliga a ofrecer el código completo a quienes usan el sistema, aunque sea
    por la red.

  Conviene que lo revise Legales.
- La alternativa sin costo es **RF-DETR (Apache-2.0)**. El detector está detrás de una interfaz chica
  (`app/detector.py`), así que cambiarlo no toca el resto.
- Los modelos faciales (YuNet MIT, SFace Apache-2.0) y mediamtx (MIT) no tienen ese problema.

**Legal y personas**
- **Hay que pasarlo por Legales, RRHH y, si corresponde, el gremio antes de instalar cámaras de análisis.**
- La imagen y sobre todo la **biometría** son datos personales; la cara es dato sensible (Ley 25.326, Res. AAIP 4/2019).
- En una relación laboral el consentimiento es frágil, y una cámara también procesa a quienes no se registraron.
- Para producción propongo **ocupación anónima por cámara e identidad desde el sistema**: quién inició la orden de
  trabajo en Odoo MRP, o una tarjeta o un molinete. Da el mismo dato de "quién estuvo" sin biometría.
- Definir **retención**: cuánto tiempo se guardan visitas y alertas. Las posiciones del mapa de calor hoy se guardan 30 días.
- Informar con cartelería que hay análisis de video.

**Cámaras y calibración**
- Ubicación pensada para cada objetivo:
  - Para ocupación y tiempos, cámara en altura que vea la estación completa, con anclaje `feet`.
  - Para identificación, cámara a la altura de la cara en un punto de paso (ver [Qué se midió](#qué-se-midió)).
- Ajustar por estación los umbrales de alertas, la gracia de salida y el ángulo de flexión. Lo que sirve para lijado no
  sirve para el depósito.
- **EPP** (casco, antiparras, protección auditiva) no está incluido: no hay un modelo confiable listo. Requiere
  entrenar uno con imágenes propias de la planta.

---

## Licencias de terceros

- Ultralytics (AGPL-3.0), PyTorch (BSD) y OpenCV con los modelos YuNet (MIT) y SFace (Apache-2.0).
- FastAPI (MIT) y mediamtx (MIT).
- `webcam.bat` baja la build *essentials* de FFmpeg de gyan.dev (GPL), que no se redistribuye con este proyecto.
- El video de prueba `vtest.avi` es de los ejemplos de OpenCV.
- LFW y UR Fall Detection Dataset se usaron solo para medir y no se incluyen en el repositorio.
