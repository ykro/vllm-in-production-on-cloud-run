# Scripts de Cloud Run

Tres scripts, sin Dockerfile. El servicio usa la imagen oficial `vllm/vllm-openai:v0.33.0` tal como se publica y toda la configuración va en `--args`. El proyecto se pasa siempre por la variable `GCP_PROJECT_ID`; ningún script trae un id escrito.

| Script | Qué hace | Cuándo |
| :--- | :--- | :--- |
| `stage_weights.sh` | Crea el bucket `${GCP_PROJECT_ID}-vllm-weights` en la misma región que la GPU y copia los pesos de Hugging Face con Cloud Build. Si ya están, no hace nada. | Una vez |
| `deploy_vllm.sh` | Despliega `vllm-gemma` con 1 L4 por instancia, pesos montados en solo lectura en `/models`, startup probe de 1500 s y liveness probe en `/health`. | Cada vez que cambias algo |
| `teardown.sh` | Borra el servicio. Con `DELETE_WEIGHTS=1` borra también el bucket. | Al terminar |

## Variables

| Variable | Default | Para qué |
| :--- | :--- | :--- |
| `GCP_PROJECT_ID` | (obligatoria) | Tu proyecto de Google Cloud |
| `GCP_REGION` | `us-east4` | Región con cuota de L4 en Cloud Run |
| `WEIGHTS_SOURCE` | `bucket` | `hf` descarga de Hugging Face en cada cold start (para comparar en el experimento 01) |
| `MIN_INSTANCES` | `0` | `0` es on-demand y no cobra en reposo; `1` cobra la L4 las 24 horas |
| `MAX_INSTANCES` | `1` | `2` para el experimento 04 (necesita cuota para una segunda L4) |
| `CONCURRENCY` | `16` | Requests por instancia antes de que Cloud Run arranque otra |

## Por qué los pesos van en un bucket

En la corrida real (vLLM 0.33.0, `google/gemma-4-E4B-it`, 1 L4) el cold start con pesos de Hugging Face tardó unos 8.5 minutos y la descarga anónima se llevó unos 262 s. En otra región esa descarga no terminó en 15 minutos. Con el bucket en la misma región, el cold start ya no depende de Hugging Face.

El montaje usa la cuenta de servicio del servicio de Cloud Run, que necesita permiso de lectura sobre el bucket. Revisa los permisos en tu proyecto antes de desplegar; estos scripts no cambian IAM.

## Flujo

```bash
export GCP_PROJECT_ID=tu-proyecto
bash gcp/stage_weights.sh
MAX_INSTANCES=2 bash gcp/deploy_vllm.sh
gcloud run services proxy vllm-gemma --region us-east4 --port 8000   # otra terminal
make logs            # guarda data/startup_log.txt (el cold start del despliegue) para el 01
make verify-live
bash gcp/teardown.sh
```
