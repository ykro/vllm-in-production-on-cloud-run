#!/usr/bin/env bash
# Deletes the Cloud Run service. With min-instances 0 an idle service costs nothing, but with
# min-instances 1 it bills the L4 around the clock until you delete it or set it back to 0.
# The weights bucket (about 16 GB of storage, billed monthly) stays unless DELETE_WEIGHTS=1.
#
#   GCP_PROJECT_ID=my-project bash gcp/teardown.sh
#   GCP_PROJECT_ID=my-project DELETE_WEIGHTS=1 bash gcp/teardown.sh
set -euo pipefail
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to your Google Cloud project id}"
REGION="${GCP_REGION:-us-east4}"
SERVICE="${VLLM_SERVICE:-vllm-gemma}"
BUCKET="${WEIGHTS_BUCKET:-${PROJECT_ID}-vllm-weights}"

gcloud run services delete "${SERVICE}" --project "${PROJECT_ID}" --region "${REGION}" --quiet

if [[ "${DELETE_WEIGHTS:-0}" == "1" ]]; then
  gcloud storage rm --recursive "gs://${BUCKET}" --project "${PROJECT_ID}"
fi
