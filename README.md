# vLLM en Producción sobre Cloud Run: Cold Start, Métricas, Autoscaling y Costo Real

- **Por qué es relevante:** levantar `vllm serve` una vez es la parte fácil. Dejarlo corriendo para otras personas pide responder cuatro preguntas que una demo esconde: cuánto tarda en arrancar una instancia nueva, cómo sabe Cloud Run que ya está lista, qué números mirar mientras atiende tráfico y cuánto cuesta cada millón de tokens comparado con pagar una API. Este repo responde las cuatro con mediciones.
- **Para quién es:** desarrolladores que saben Python, ya llamaron alguna API de IA y ya levantaron su primer servidor con Gemma en [Tu primer servidor vLLM con Gemma](https://github.com/ykro/first-vllm-server-with-gemma). No hace falta JAX ni PyTorch.
- **Quién debería verlo:** quien tiene que decidir entre servir un modelo abierto propio y pagar por token a un proveedor, o quien ya lo sirve y quiere saber por qué la primera petición del día tarda minutos.
- **Serie completa:** quinta unidad del Field Guide de vLLM (servir modelos abiertos con vLLM). Antes van [Tu primer servidor vLLM con Gemma](https://github.com/ykro/first-vllm-server-with-gemma) y el índice, [vLLM for Software Devs Field Guide](https://github.com/ykro/vllm-for-software-devs-field-guide).

La pregunta de esta unidad: **¿cuánto cuesta de verdad, y qué hace falta para dejarlo corriendo?**

---

## La historia que motiva el repo

Con vLLM 0.33.0, `google/gemma-4-E4B-it` y una GPU L4 en Cloud Run, descargando los pesos de Hugging Face de forma anónima en cada arranque, un cold start tardó **unos 8.5 minutos**:

| Fase | Tiempo aproximado |
| :--- | :--- |
| Descargar los pesos de Hugging Face | ~262 s |
| `torch.compile` | ~55 s |
| Captura de CUDA graphs | ~10 s |
| Contenedor, carga a la GPU, perfil del KV-cache y arranque del API | el resto |

En otra región la descarga anónima ni siquiera terminó en 15 minutos. Por eso aquí los pesos se copian una sola vez a un bucket de Cloud Storage en la misma región que la GPU y se montan en solo lectura (`gcp/stage_weights.sh`). Y por eso `min-instances` deja de ser un detalle de configuración: con `0` no pagas nada en reposo pero el primer usuario espera minutos; con `1` nadie espera pero la L4 se cobra las 24 horas.

---

## Los cinco experimentos

| # | Archivo | Qué responde |
| :--- | :--- | :--- |
| 01 | `experiments/01_cold_start_anatomy.py` | ¿En qué se van los minutos del cold start? Lee el log de arranque exportado de Cloud Logging y lo parte en fases (contenedor, descarga, carga a la GPU, `torch.compile`, perfil del KV-cache, CUDA graphs, API, startup probe). Las fases se miden entre eventos del log, así que siempre suman el total. Con dos logs compara Hugging Face contra el bucket fase por fase. |
| 02 | `experiments/02_health_and_readiness.py` | ¿Está vivo, qué sirve y puede generar? `/health`, `/v1/models`, `/version` y una completion de un solo token como prueba de que está listo. Explica el startup probe de Cloud Run: el presupuesto es `periodSeconds x failureThreshold` (10 x 150 = 1500 s) y tiene que superar al cold start más lento que hayas medido. |
| 03 | `experiments/03_metrics_dashboard.py` | ¿Qué miro en `/metrics`? De cientos de series Prometheus imprime las que importan: requests corriendo y en espera, uso del KV-cache, TTFT y latencia de punta a punta (p50/p95 desde los histogramas, solo para la ráfaga medida) y tokens generados. Busca los nombres por fragmentos porque vLLM los renombra entre versiones. |
| 04 | `experiments/04_autoscaling_two_instances.py` | ¿Qué pasa con más usuarios de los que aguanta una GPU? Un generador de carga mantiene más requests en vuelo de los que acepta una instancia; Cloud Run arranca una segunda (con `MAX_INSTANCES=2` y cuota) y el experimento registra la latencia de cada request en el tiempo. Identifica la instancia por un header cuando existe y, si no, por el `process_start_time_seconds` de cada scrape de `/metrics`. |
| 05 | `experiments/05_cost_vs_api.py` | ¿Cuándo conviene la GPU propia? Costo por millón de tokens de salida con 1 y con N usuarios, costo en reposo con `min-instances` 0 y 1, comparación con el precio de lista de una API Gemini Flash y el break-even en tokens por día. |

El juez `verify_production.py` corre los cinco y revisa comportamientos, no cifras exactas: que las fases sumen el total, que el servidor esté listo y el presupuesto del probe alcance, que `/metrics` se pueda leer, que la segunda instancia baje el p95 bajo carga y que el break-even exista.

---

## Cómo correrlo

### Sin GPU (`make all`)

vLLM necesita una GPU NVIDIA y no corre en una Mac. Para revisar la lógica sin GPU, cada experimento acepta `--mock`, que levanta un servicio falso con la misma API HTTP y un autoscaler simulado: cada instancia acepta 8 requests a la vez y una segunda instancia aparece un segundo y medio después de que se forma fila. Sus números son inventados a propósito.

```bash
make all      # ruff + pytest + verify_production.py --mock
make report   # el juez en mock con el resultado completo de cada experimento
```

Solo usa la biblioteca estándar de Python y el SDK de `openai`.

### Contra Cloud Run (`make verify-live`)

```bash
export GCP_PROJECT_ID=tu-proyecto
make stage                         # una vez: pesos al bucket en us-east4
MIN_INSTANCES=1 MAX_INSTANCES=2 make deploy   # 1 L4 por instancia, hasta 2; MIN_INSTANCES=1 solo mientras mides
gcloud run services proxy vllm-gemma --region us-east4 --port 8000   # en otra terminal
make logs REVISION=<revisión>      # guarda el arranque de esa revisión en data/startup_log.txt
make verify-live
make cool                          # min-instances 0 y max-instances 1 otra vez
make teardown
```

El túnel firma cada request con tu identidad de `gcloud`, así que el servicio no tiene URL pública ni API key en el código. El experimento 04 en vivo mantiene 48 usuarios durante 15 minutos para dar tiempo a que la segunda instancia termine su propio cold start; mientras corre, la consola de Cloud Run muestra el número de instancias. Detalle de los scripts y sus variables en [`gcp/README.md`](gcp/README.md).

---

## Resultados

Corrida en vivo del 2026-11-11 con `MIN_INSTANCES=1 MAX_INSTANCES=2 make deploy`, `make logs REVISION=<revisión>` y `make verify-live` (vLLM 0.33.0, `google/gemma-4-E4B-it` en bf16, Cloud Run `us-east4`, una NVIDIA L4 por instancia, pesos montados desde un bucket, concurrencia 16). Todo queda en `benchmarks/latest_results.json`, `benchmarks/terminal_output.txt`. El log completo del arranque (`data/startup_log.txt`) no se versiona porque trae el nombre de tu proyecto; `data/startup_log_v033_excerpt.txt` es el extracto con las líneas que lee el experimento 01. El juez pasa los 5 checks.

| Experimento | Resultado en L4 |
| :--- | :--- |
| **01. Anatomía del cold start** | **580 s (9.67 min)** de `Starting new instance` a la probe exitosa: contenedor y montaje 65.5 s, arranque del proceso 86.3, pesos a la GPU **262.9** (vLLM reporta 263.3), preparar el modelo 57.5, `torch.compile` 53.7 (vLLM reporta 53.73), CUDA graphs 22.4, KV-cache 0.8, resto del motor 17.4, API 5.9, probe 8.0 |
| **02. Salud y listo** | `/health` 200 en 266 ms, `/v1/models` y `/version` responden; una completion de 1 token en 2,275 ms. Presupuesto de la startup probe 1,500 s |
| **03. `/metrics`** | 450 series; las 4 que importan encontradas por fragmentos (`vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc`, `vllm:generation_tokens_total`). Ráfaga de 8 peticiones: 8 corriendo, 0 en cola, KV-cache al 1.1 %, 512 tokens; TTFT p50 0.15 s y p95 0.24 s; latencia total p50 3.75 s y p95 4.88 s |
| **04. Segunda instancia bajo carga** | 48 usuarios durante 900 s: 3,889 peticiones, 0 errores. Cloud Run arrancó la segunda instancia a los 66 s de empezar la carga; empezó a atender cerca del segundo **600** (el script la vio en las métricas a los 687 s). Antes, latencia p50 de unos 14 s y p95 de 15.7 s; después, p50 de unos 7 s y p95 de 10.2 s (1.53 veces menos), y las peticiones por ventana de 30 s pasaron de unas 104 a unas 200 |
| **05. Costo contra API** | 1 usuario 25.3 tokens/s, **USD 15.58** por millón de tokens de salida; 8 usuarios 197.0 tokens/s, **USD 2.00**. API `gemini-3.8-flash` con 2 tokens de entrada por cada token de salida: USD 5.25 hasta fin de 2026 y USD 10.50 desde 2027. Punto de equilibrio **6,495,543 tokens por día** (38 % de lo que produce una L4 llena); `min-instances 1` cuesta USD 1,037 al mes |

Salida del juez en vivo:

```text
01. Cold start anatomy           | 9.67 min: weights 263 s, compile 54 s        | graphs 22 s                  | PASS
02. Health and readiness         | /health 200, 1 token in 2275 ms              | probe budget 1500 s          | PASS
03. /metrics that matter         | TTFT p95 0.24 s, e2e p95 4.875 s             | 512 tokens                   | PASS
04. Second instance under load   | 2 instances, 2nd at 687.23 s                 | p95 15.6527->10.2099 s       | PASS
05. Cost vs hosted API           | $15.58 (1u) $2.00 (8u) /1M                   | break-even 6,495,543 tok/day | PASS
```

### Lo que se midió y cambió el código

- **El log real no tenía el formato de la muestra.** vLLM 0.33.0 escribe `Model loading took ... and 263.25 seconds`, `torch.compile took` (la muestra decía `takes`), `Loading model from scratch...`, y registra los CUDA graphs antes del tamaño del KV-cache. El experimento 01 ahora arma las fases ordenando los eventos por hora, y sus cifras coinciden con las que imprime vLLM. Con los pesos en el bucket no existe la línea de descarga: el montaje con GCS FUSE lee el archivo mientras el motor lo carga (los 263 s).
- **El escalado con GPU responde en minutos.** La segunda instancia pagó su propio cold start (unos 9 minutos desde que Cloud Run la pidió). Mientras tanto los 48 usuarios esperaron unos 14 s por respuesta.
- **`make cool` es otro despliegue.** Cambiar `min-instances` crea una revisión nueva y Cloud Run arranca una instancia para comprobarla; la GPU sigue encendida unos 10 minutos más antes de que todo vuelva a cero. En la corrida de la unidad anterior, la revisión nueva tardó de las 17:16 a las 17:25 UTC en estar lista.
- **Cuota de GPU.** La segunda L4 sí estuvo disponible en `us-east4` durante la corrida (la cuota regional es de 3 y se comparte con otros servicios de la cuenta).

### Precios usados

| Qué | Precio de lista | Revisado | Fuente |
| :--- | :--- | :--- | :--- |
| Cloud Run, 1 L4 + 8 vCPU + 32 GiB | 1.4209 USD por hora | 2026-11-10 | [cloud.google.com/run/pricing](https://cloud.google.com/run/pricing) |
| `gemini-3.8-flash`, hasta el 31-dic-2026 | 0.75 USD entrada / 3.75 USD salida por millón de tokens | 2026-11-10 | [aistudio.google.com/docs/pricing](https://aistudio.google.com/docs/pricing) |
| `gemini-3.8-flash`, desde el 1-ene-2027 | 1.50 USD entrada / 7.50 USD salida por millón de tokens | 2026-11-10 | misma página |

Los dos archivos están en `benchmarks/pricing.json` y `benchmarks/api_prices.json`. Cambian por región y con el tiempo; revísalos antes de decidir.

### Cómo leer el costo

- **Por token y por día son preguntas distintas.** Con 8 usuarios a la vez la misma L4 escribe varias veces más tokens por hora que con uno, así que el costo por millón baja mucho. Ese número solo vale si la GPU está ocupada; una instancia mínima encendida cuesta lo mismo por hora con tráfico o sin él.
- **La API cobra también la entrada.** El experimento 05 suma 2 tokens de entrada por cada token de salida (configurable con `--input-per-output`). En tu GPU el prefill ya está dentro del precio por hora.
- **Break-even.** Es el número de tokens de salida por día a partir del cual `min-instances 1` (la L4 las 24 horas) cuesta menos que la API. El experimento también dice qué fracción de la capacidad de una instancia representa; si pasa del 100 %, una sola L4 no alcanza.
- **`min-instances 0`.** No cobra en reposo, pero cada vez que despierta paga el cold start más el tiempo que Cloud Run mantiene la instancia sin tráfico antes de apagarla. Cloud Run admite hasta 15 minutos; en la corrida, la revisión sin tráfico que dejó `make cool` se apagó 614 s (10.2 min) después de pasar su startup probe (`data/instance_lifecycle_log.txt`). Con los 580 s de arranque, el experimento reporta unos **USD 0.47 por despertar**.
- **Supuestos por confirmar.** Que una instancia mínima inactiva con GPU se cobre a la tarifa completa sigue marcado así en `benchmarks/pricing.json`: no se vio en una factura. Los 10.2 minutos antes de apagarse salen de una sola instancia; Cloud Run no promete ese tiempo. Los nombres de las métricas de `/metrics` sí se confirmaron en vLLM 0.33.0. El presupuesto de la startup probe (1,500 s) da 2.58 veces los 580 s medidos; `verify_production.py` le pasa al experimento 02 el cold start que mide el 01.

---

## Estructura del repositorio

```text
.
├── benchmarks/
│   ├── pricing.json                     # Precio de lista de la instancia de Cloud Run con L4
│   └── api_prices.json                  # Precio de lista de la API Gemini Flash, con fecha y fuente
├── data/
│   ├── instance_lifecycle_log.txt       # Arranques, probes y apagados de las instancias (tiempo antes de apagarse)
│   ├── startup_log.txt                  # (no versionado) log completo de make logs
│   ├── startup_log_v033_excerpt.txt     # Extracto del log real de vLLM 0.33.0 (lo usa un test)
│   └── startup_log_sample.txt           # Log de ejemplo con el formato anterior; lo lee --mock
├── diagrams/                            # 3 diagramas blue-gold (html, svg, png)
├── experiments/
│   ├── 01_cold_start_anatomy.py         # Fases del cold start desde el log; Hugging Face vs bucket
│   ├── 02_health_and_readiness.py       # /health, /v1/models, /version, readiness y startup probe
│   ├── 03_metrics_dashboard.py          # Las series de /metrics que importan, con p50/p95
│   ├── 04_autoscaling_two_instances.py  # Carga que obliga a Cloud Run a arrancar otra instancia
│   └── 05_cost_vs_api.py                # Costo por millón de tokens, reposo y break-even
├── gcp/
│   ├── README.md
│   ├── stage_weights.sh                 # Pesos de Hugging Face a un bucket en la misma región
│   ├── deploy_vllm.sh                   # Imagen oficial de vLLM en Cloud Run, sin Dockerfile
│   └── teardown.sh                      # Borra el servicio (y el bucket si lo pides)
├── tests/
│   └── test_production.py               # pytest contra el servicio falso
├── vllm_lab/
│   ├── cli.py                           # --mock, --base-url y el aviso del túnel
│   ├── client.py                        # Cliente del SDK de OpenAI apuntando a tu vLLM
│   ├── mock_server.py                   # vLLM falso con autoscaler simulado
│   └── prom.py                          # Lector mínimo del formato de texto de Prometheus
├── Makefile
├── pyproject.toml
└── verify_production.py                 # El Outside Judge
```

---

## Licencia

MIT.
