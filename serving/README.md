# Serving

The AVM behind an HTTP interface. Two things live here.

| | |
|---|---|
| `artifacts/` | The **promoted** model: the fitted pipeline, the comparables pool, and the model card. Committed, so the container needs neither the network nor the notebooks. |
| `../src/ames/service.py` | The application. |

The service runs locally and in any container host. It is not hosted anywhere public;
see "Why there is no hosted copy" below.

## Why promotion is a separate step

The notebooks write into `data/processed/`, which is gitignored and rebuilt by
`make all`. A container cannot run that at build time: it would need the network and
several minutes, and it would retrain on every deploy.

Serving therefore reads a snapshot, produced by an explicit step:

```
make promote
```

This keeps the deployed model a separate, auditable object from the model that was last
trained. If the two diverge, only `make promote` can have closed the gap, and the diff
shows it.

## Running it

```
make serve                      # local, with reload
docker build -t ames-avm .      # from the project root
docker run --rm -p 7860:7860 ames-avm
```

Then open `http://localhost:7860/docs`.

## Why there is no hosted copy

This repository used to deploy the container to a Hugging Face Space. It no longer
does: Hugging Face now serves Gradio and Docker Spaces only on a **paid** tier, and
`repos create --sdk docker` returns `402 Payment Required` on a free account. Only
*static* Spaces remain free.

A paid subscription to keep a demo running is not worth it for a portfolio project, so
the hosted surface is now a static page built by `make site`: the numbers, the figures
and three worked valuation reports, generated from the pipeline's own outputs. Only the
hosted copy of the API was lost; the service itself is unchanged and runs with the
commands above.

Any container host runs the root `Dockerfile` unmodified. For a return to Hugging Face,
note that `huggingface-cli` was retired in favour of `hf`, and the old binary is now a
stub that refuses to run.

## What the service will not do

It will not value a property in an unknown neighbourhood; that request returns a `422`
with the list of valid neighbourhoods instead of a city-wide guess. It will not return a
point estimate without an interval. Every response carries the disclaimer, because an
easily called endpoint will sooner or later have its output pasted into a decision.
