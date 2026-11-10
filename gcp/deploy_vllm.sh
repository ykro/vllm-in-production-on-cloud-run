#!/usr/bin/env bash
# Deploys the official vLLM image with Gemma on Cloud Run with one NVIDIA L4 GPU per instance.
# No Dockerfile: the image is vllm/vllm-openai as published, and every setting goes in --args.
#
# Weights come from a Cloud Storage bucket in the same region, mounted read-only at /models
# (run gcp/stage_weights.sh once). Measured with weights pulled anonymously from Hugging Face,
# a cold start took about 8.5 min, ~262 s of it downloading; in another region the download
# did not finish within 15 min. The bucket takes Hugging Face out of the cold start.
#
#   GCP_PROJECT_ID=my-project bash gcp/deploy_vllm.sh                   # 1 instance, scales to 0
#   GCP_PROJECT_ID=my-project MAX_INSTANCES=2 bash gcp/deploy_vllm.sh   # for experiment 04
#   GCP_PROJECT_ID=my-project WEIGHTS_SOURCE=hf bash gcp/deploy_vllm.sh # compare cold starts
#
# Then open a local tunnel (no public URL, no API key in your code):
#   gcloud run services proxy vllm-gemma --region us-east4 --port 8000
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-east4}"            # a region with Cloud Run L4 quota
SERVICE="${VLLM_SERVICE:-vllm-gemma}"
IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai:v0.33.0}"
MODEL="${VLLM_MODEL:-google/gemma-4-E4B-it}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
BUCKET="${WEIGHTS_BUCKET:-${PROJECT_ID}-vllm-weights}"
WEIGHTS_SOURCE="${WEIGHTS_SOURCE:-bucket}"  # bucket | hf
MIN_INSTANCES="${MIN_INSTANCES:-0}"         # 0 = on-demand, nothing billed while idle
MAX_INSTANCES="${MAX_INSTANCES:-1}"         # 2 lets experiment 04 see a second L4 (needs quota)
CONCURRENCY="${CONCURRENCY:-16}"            # requests per instance before Cloud Run scales out

SOURCE_FLAGS=()
if [[ "${WEIGHTS_SOURCE}" == "bucket" ]]; then
  if ! gcloud storage ls "gs://${BUCKET}/${MODEL}/config.json" --project "${PROJECT_ID}" \
      >/dev/null 2>&1; then
    echo "No weights in gs://${BUCKET}/${MODEL}. Run gcp/stage_weights.sh first," >&2
    echo "or WEIGHTS_SOURCE=hf to download from Hugging Face on every cold start." >&2
    exit 1
  fi
  SOURCE_FLAGS=(--add-volume "name=weights,type=cloud-storage,bucket=${BUCKET},readonly=true"
                --add-volume-mount "volume=weights,mount-path=/models")
  MODEL_ARG="/models/${MODEL}"
  echo "Weights from gs://${BUCKET}/${MODEL} (read-only mount)"
else
  MODEL_ARG="${MODEL}"
  echo "Weights from Hugging Face (${MODEL}), downloaded on every cold start"
fi

# Startup probe budget: periodSeconds x failureThreshold = 10 x 150 = 1500 s, comfortably
# above the slowest cold start measured (experiment 02 explains the arithmetic).
gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" --region "${REGION}" \
  --image "${IMAGE}" \
  --args="--model=${MODEL_ARG},--served-model-name=gemma,--max-model-len=${MAX_LEN},--port=8080,--host=0.0.0.0" \
  --port 8080 \
  ${SOURCE_FLAGS[@]+"${SOURCE_FLAGS[@]}"} \
  --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy \
  --cpu 8 --memory 32Gi --no-cpu-throttling \
  --min-instances "${MIN_INSTANCES}" --max-instances "${MAX_INSTANCES}" \
  --concurrency "${CONCURRENCY}" --timeout 900 \
  --startup-probe "httpGet.path=/health,httpGet.port=8080,periodSeconds=10,failureThreshold=150,timeoutSeconds=5" \
  --liveness-probe "httpGet.path=/health,httpGet.port=8080,periodSeconds=30,failureThreshold=3,timeoutSeconds=5" \
  --no-allow-unauthenticated

gcloud run services describe "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" \
  --format 'value(status.url)'
