# vLLM en Producción sobre Cloud Run: Cold Start, Métricas, Autoscaling y Costo Real

- **Por qué es relevante:** levantar `vllm serve` una vez es la parte fácil. Dejarlo corriendo para otras personas pide responder cuatro preguntas que una demo esconde: cuánto tarda en arrancar una instancia nueva, cómo sabe Cloud Run que ya está lista, qué números mirar mientras atiende tráfico y cuánto cuesta cada millón de tokens comparado con pagar una API. Este repo responde las cuatro con mediciones.
- **Para quién es:** desarrolladores que saben Python, ya llamaron alguna API de IA y ya levantaron su primer servidor con Gemma en C1. No hace falta JAX ni PyTorch.
- **Quién debería verlo:** quien tiene que decidir entre servir un modelo abierto propio y pagar por token a un proveedor, o quien ya lo sirve y quiere saber por qué la primera petición del día tarda minutos.
- **Serie completa:** quinta unidad de la Serie C (servir modelos abiertos con vLLM). Antes van [C1 · Tu primer servidor vLLM con Gemma](https://github.com/ykro/first-vllm-server-with-gemma) y el índice de la serie, [vLLM for Software Devs Field Guide](https://github.com/ykro/vllm-for-software-devs-field-guide).

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
MAX_INSTANCES=2 make deploy        # 1 L4 por instancia, hasta 2 instancias
gcloud run services proxy vllm-gemma --region us-east4 --port 8000   # en otra terminal
make logs                          # guarda el arranque en data/startup_log.txt
make verify-live
make teardown
```

El túnel firma cada request con tu identidad de `gcloud`, así que el servicio no tiene URL pública ni API key en el código. El experimento 04 en vivo mantiene 48 usuarios durante 15 minutos para dar tiempo a que la segunda instancia termine su propio cold start; mientras corre, la consola de Cloud Run muestra el número de instancias. Detalle de los scripts y sus variables en [`gcp/README.md`](gcp/README.md).

---

## Resultados

**Pendiente de la corrida en vivo.** Esta sección se llena con `benchmarks/terminal_output.txt` y `benchmarks/latest_results.json` cuando `make verify-live` corra contra Cloud Run. Lo único medido hasta ahora es la historia del cold start de arriba; `data/startup_log_sample.txt` es un log de ejemplo escrito a partir de esas duraciones para que el experimento 01 tenga algo que leer sin desplegar nada.

Salida del juez en modo mock, solo para ver la forma del reporte (las cifras de 02 a 05 son del servicio falso):

```text
01. Cold start anatomy           | 8.5 min: weights 274 s, compile 55 s         | graphs 10 s                  | PASS
02. Health and readiness         | /health 200, 1 token in 14 ms                | probe budget 1500 s          | PASS
03. /metrics that matter         | TTFT p95 0.0243 s, e2e p95 1.475 s           | 512 tokens                   | PASS
04. Second instance under load   | 2 instances, 2nd at 1.608 s                  | p95 1.0235->0.6237 s         | PASS
05. Cost vs hosted API           | $6.51 (1u) $0.94 (8u) /1M                    | break-even 6,495,543 tok/day | PASS
```

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
- **`min-instances 0`.** No cobra en reposo, pero cada vez que despierta paga el cold start más el tiempo que Cloud Run mantiene la instancia sin tráfico antes de apagarla (hasta 15 minutos). El experimento lo reporta como costo por despertar.
- **Supuestos por confirmar.** Que una instancia mínima inactiva con GPU se cobre a la tarifa completa y el tiempo exacto antes de apagarse están marcados así en `benchmarks/pricing.json` hasta verlos en la factura de la corrida en vivo. Los nombres de las métricas de `/metrics` en vLLM 0.33.0 también se confirman en esa corrida.

---

## Estructura del repositorio

```text
.
├── benchmarks/
│   ├── pricing.json                     # Precio de lista de la instancia de Cloud Run con L4
│   └── api_prices.json                  # Precio de lista de la API Gemini Flash, con fecha y fuente
├── data/
│   └── startup_log_sample.txt           # Log de arranque de ejemplo, escrito desde la corrida real
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
