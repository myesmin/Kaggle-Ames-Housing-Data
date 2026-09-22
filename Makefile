# Ames Housing -> Real-Estate Valuation & Investment Risk Platform
#
#   make setup   create .venv (Python 3.12) and install requirements
#   make data    download every external source and assemble data/processed/
#   make test    run the unit, leakage and data-contract tests
#   make run     execute all ten notebooks end to end, in order
#   make map     regenerate the notebook dependency diagram
#   make promote snapshot the trained model into serving/artifacts/
#   make site    build the static results site into site/
#   make serve   run the valuation API locally on :7860
#   make image   build the serving container
#   make all     setup -> data -> test -> run
#
# Everything runs out of ./.venv -- never the anaconda base environment.

VENV    := .venv
PY      := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
NBCONV  := $(VENV)/bin/jupyter nbconvert
NOTEBOOKS := $(sort $(wildcard notebooks/*.ipynb))

.PHONY: all setup data test run map promote site serve image clean clean-data figures help

all: setup data test run map

help:
	@grep -E '^#   make' $(MAKEFILE_LIST) | sed 's/^#   /  /'

$(VENV)/bin/python:
	uv venv --python 3.12 $(VENV)

setup: $(VENV)/bin/python requirements.txt pyproject.toml
	uv pip install --python $(VENV) -r requirements.txt
	uv pip install --python $(VENV) -e .
	@echo "environment ready: $$($(PY) -V)"

data:
	$(PY) -m ames.data

data-force:
	$(PY) -m ames.data --force

test:
	$(PYTEST) tests/

# Notebooks are executed in filename order and written back in place, so the rendered
# outputs in the repo are always the ones the current code actually produced.
run: $(NOTEBOOKS)
	@for nb in $(NOTEBOOKS); do \
		echo "==> $$nb"; \
		$(NBCONV) --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=1800 \
			--ExecutePreprocessor.kernel_name=python3 "$$nb" || exit 1; \
	done
	@echo "all notebooks executed cleanly"

map:
	$(PY) tools/flow_map.py

# --- serving ---------------------------------------------------------------
# `promote` snapshots the trained model into serving/artifacts/ (committed), which is
# what the container reads.  Deliberately manual: deploying a model should be a
# decision, not a side effect of running a notebook.
promote:
	$(PY) tools/promote_model.py

# The static results site.  Reads the same summary JSONs the notebooks write, so the
# published page cannot drift from the pipeline that produced it.
site:
	$(PY) tools/build_site.py

serve: serving/artifacts/avm_model.joblib
	$(VENV)/bin/uvicorn ames.service:app --reload --port 7860

serving/artifacts/avm_model.joblib:
	@echo "no promoted model -- run 'make promote' (needs 'make run' first)" && exit 1

image:
	docker build -t ames-avm .
	@echo "run it:  docker run --rm -p 7860:7860 ames-avm" 

figures:
	@ls -la reports/figures

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ src/*.egg-info

# Deliberately separate from `clean`: re-downloading everything takes a few minutes and
# is rarely what you want.
clean-data:
	rm -rf data/processed/* data/external/* data/raw/AmesHousing.txt data/raw/ames_geo.rda
	rm -rf reports/valuations
