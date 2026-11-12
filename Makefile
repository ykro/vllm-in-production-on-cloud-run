PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)
REGION ?= us-east4
SERVICE ?= vllm-gemma

.PHONY: all verify verify-live report test lint stage deploy warm cool tunnel logs teardown mock clean

# No GPU needed: lint, tests and the judge against the built-in mock service
all: lint test verify

verify:
	$(PYTHON) verify_production.py --mock

# Real numbers: needs the tunnel open (make tunnel) in another terminal.
# Experiment 04 keeps 48 users busy for 15 min; deploy with MAX_INSTANCES=2 first.
verify-live:
	$(PYTHON) verify_production.py

report:
	$(PYTHON) verify_production.py --mock --verbose

test:
	$(PYTEST) -q

lint:
	$(RUFF) check .

stage:
	bash gcp/stage_weights.sh

deploy:
	bash gcp/deploy_vllm.sh

# Keep one instance warm while you work (no cold start between experiments); scale to zero after.
# `cool` also puts max-instances back to 1 if you deployed with MAX_INSTANCES=2.
warm:
	gcloud run services update $(SERVICE) --region $(REGION) --project $(GCP_PROJECT_ID) --min-instances 1

cool:
	gcloud run services update $(SERVICE) --region $(REGION) --project $(GCP_PROJECT_ID) --min-instances 0 --max-instances 1

tunnel:
	gcloud run services proxy $(SERVICE) --region $(REGION) --port 8000

# Saves the last hour of the service's logs for experiment 01 (newest first, as gcloud prints)
logs:
	gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="$(SERVICE)"$(if $(REVISION), AND resource.labels.revision_name="$(REVISION)")' \
	  --project "$${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}" --freshness 1h --limit 5000 \
	  --format 'value(timestamp,textPayload)' > data/startup_log.txt
	@echo "Saved data/startup_log.txt"

teardown:
	bash gcp/teardown.sh

mock:
	$(PYTHON) -m vllm_lab.mock_server

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
