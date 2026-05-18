[[MLOps]] is the discipline of making [[Machine Learning]] systems reproducible, deployable, observable, and governable.

It is not "Kubernetes for notebooks." The core problem is that ML systems depend on three moving things:
- Code
- Data
- Model artifacts

In normal backend development, you can usually reproduce a build from source code + dependencies. In ML, the result also depends on training data, feature definitions, labels, random seeds, hardware, library versions, hyperparameters, and sometimes nondeterministic GPU kernels. MLOps is the machinery that keeps that from becoming folklore.

Big idea: ==treat models as production artifacts in a software supply chain, but gate them on data quality and evals, not just unit tests.==


# The Classic MLOps Loop

```
Raw Data -> Features -> Train -> Evaluate -> Register -> Deploy -> Monitor -> Collect Feedback
   |          |          |         |           |          |          |             |
Parquet    Feast      PyTorch   Evals      MLflow     KServe    Evidently      Labels
dbt        dbt        Ray       W&B        W&B        BentoML   Arize          Retraining
Spark      SQL        GPUs      Great Exp  SageMaker  Triton    Langfuse       Data flywheel
```

At every step, you want lineage:

```
model_version = f(code_sha, dataset_version, feature_defs, params, environment)
```

If you cannot answer "what exact data and code produced the model currently serving production traffic?", you do not really have MLOps.


# Why MLOps is Harder than DevOps

- Data changes underneath you. The same code can produce a worse model next month.
- Ground truth is often delayed. You may not know whether a prediction was correct until days/weeks later.
- Offline metrics lie. AUC, F1, BLEU, or eval scores can improve while the product gets worse.
- Training/serving skew is common. Features computed in batch for training differ from features computed online for inference.
- Failures are often silent. The API returns `200 OK`, but the model is confidently wrong.
- Reproducibility is fragile. Notebook state, package versions, hardware, seeds, and input data all matter.
- GPUs turn infrastructure mistakes into real money very quickly.


# Data Layer

MLOps starts as [[Data Engineering]].

- Object storage: [[Amazon S3|S3]], GCS, Azure Blob. Usually the source of truth for datasets and artifacts.
- File/table formats: [[Apache Parquet|Parquet]], [[Apache Iceberg]], Delta Lake, [[Lance]], sometimes HDF5/Zarr for scientific arrays.
- Compute: [[Apache Spark|Spark]] for large distributed jobs, [[DuckDB]]/[[Polars]] for medium data, [[Ray]] for Python-native distributed workloads.
- Warehouse/lakehouse: [[Snowflake]], [[Google BigQuery|BigQuery]], [[Databricks Lakehouse]], [[Amazon Redshift|Redshift]].
- Data validation: Great Expectations, Soda, Pandera, Deequ, dbt tests.
- Dataset versioning: [[DVC]], lakeFS, Delta/Iceberg time travel, W&B Artifacts, MLflow artifacts.

Trend: the [[Modern Data Stack]] won a lot of the substrate. MLOps increasingly rides on columnar storage, object stores, SQL transforms, and lakehouse catalogs instead of bespoke "ML data platforms."


# Feature Stores

Feature stores exist to solve a specific painful problem: ==the same feature must be computed consistently for offline training and low-latency online serving.==

- Offline store: historical features for training and batch scoring.
- Online store: latest feature values for low-latency inference, often Redis/Cassandra/DynamoDB/Postgres/etc.
- Registry: feature definitions, owners, metadata, freshness, lineage.
- Materialization: moving batch/streaming features into the online store.

Tools:
- [[Feast]]: default open-source feature store name to know.
- Tecton: commercial, strong enterprise adoption.
- Featureform: open-source/managed alternative.
- Databricks Feature Store / Vertex AI Feature Store / SageMaker Feature Store: cloud-native options.

Opinionated take: ==do not add a feature store until you have training-serving skew or online feature reuse.== For many teams, a table in the warehouse plus dbt tests is enough.


# Experiment Tracking

Experiment tracking records:
- Parameters
- Metrics
- Artifacts
- Dataset references
- Code version
- Environment/dependencies
- Notes and comparisons across runs

Tools:
- [[MLflow]]: the open-source default. Tracking, artifacts, model registry, evaluation, and now GenAI tracing.
- [[Weights and Biases]]: polished SaaS default for serious model development; great UX for runs, sweeps, artifacts, reports, and registry.
- Neptune, Comet, ClearML: similar category.

Rule: every training run that might matter should be queryable later. The model file alone is not enough.


# Orchestration

The pipeline turns notebooks into repeatable jobs.

- [[Apache Airflow|Airflow]]: mature, ubiquitous, good for scheduled data workflows. Heavy for pure ML iteration.
- [[Dagster]]: asset-first, strong lineage/testability model. Good if your mental model is "datasets/models are assets."
- Prefect: Pythonic workflow orchestration, good developer experience.
- Metaflow: data-scientist-friendly, originally from Netflix, nice path from local Python to production jobs.
- [[Kubeflow]] Pipelines: Kubernetes-native ML workflows. Powerful, but a platform-team tool.
- Flyte: Kubernetes-native typed workflows for data/ML, also powerful/heavy.
- Cloud-native: Vertex AI Pipelines, SageMaker Pipelines, Azure ML pipelines, Databricks Workflows.

The important distinction:
- Data orchestration asks "did the tables/materializations run?"
- ML orchestration asks "did this exact data/code/config produce an acceptable model artifact?"


# Training Infrastructure

Development usually starts in notebooks, but production training should become scripts/jobs.

- Local exploration: Jupyter, notebooks, small sampled data.
- Reproducible runtime: [[Docker]], `uv`/conda/poetry, pinned CUDA images if GPU.
- Distributed training:
	- PyTorch DDP/FSDP
	- DeepSpeed
	- Hugging Face Accelerate
	- Ray Train
	- PyTorch Lightning/Fabric
- Hyperparameter search: Ray Tune, Optuna, W&B Sweeps, Katib.
- Managed platforms: SageMaker, Vertex AI, Azure ML, Databricks.

Useful default: start with one-machine training and clean scripts. Move to distributed training only when the single-node bottleneck is real.


# Model Registry

A model registry is the promotion gate between experimentation and production.

It should store:
- Model artifact/checkpoint
- Version
- Stage/status: candidate, staging, production, archived
- Evaluation report
- Dataset and code lineage
- Input/output schema
- Dependencies/runtime
- Model card / intended use / limitations
- Approval history

Tools:
- MLflow Model Registry
- W&B Registry
- SageMaker Model Registry
- Vertex AI Model Registry
- Azure ML registry
- [[Databricks Unity Catalog]] for governed data + AI assets

The registry is not just storage. It is the control plane for "what is allowed to serve traffic?"


# Serving Patterns

There are three common deployment shapes:

## Batch Scoring

```
Input table -> prediction job -> output table
```

This is the simplest and most underrated serving pattern. Use it for recommendations, risk scores, lead scoring, churn prediction, reporting, and anything that does not need synchronous latency.

Tools: Spark, Ray, dbt + Python, Databricks Jobs, Airflow/Dagster/Prefect, warehouse UDFs in some cases.

## Online Inference

```
HTTP/gRPC request -> feature lookup -> model -> response
```

Tools:
- FastAPI: simple Python API around a model.
- BentoML: Python-native model serving and packaging.
- MLServer / Seldon: Kubernetes-oriented serving.
- KServe: standardized model serving on [[Kubernetes]].
- NVIDIA Triton: high-performance multi-framework inference, especially on GPUs.
- TorchServe / TensorFlow Serving: older framework-native servers.

You care about latency, throughput, autoscaling, model warmup, health checks, batching, schema validation, and rollback.

## LLM Serving

For LLM products, the default should be managed APIs until there is a concrete reason not to.

Self-host when you need:
- Data residency / privacy
- Cost control at high volume
- Custom fine-tunes or open-weight models
- Latency control
- Provider independence

Tools:
- vLLM: dominant open-source high-throughput LLM serving engine.
- Text Generation Inference (TGI): Hugging Face's serving stack.
- SGLang: fast serving/runtime for structured generation and agentic workloads.
- TensorRT-LLM / Triton: NVIDIA-optimized path.
- LiteLLM / OpenRouter / Vercel AI Gateway: routing/gateway layer across model providers.


# Deployment Patterns

- Shadow deployment: send production inputs to a candidate model, ignore its outputs, compare offline.
- Canary deployment: send a small slice of real traffic to the candidate.
- Blue/green: swap between two complete serving stacks.
- Champion/challenger: current production model vs candidate model.
- A/B test: compare business impact, not just model metrics.
- Rollback: promote the previous known-good model quickly.

Model rollout should be boring. The hard part is deciding whether the model is good enough.


# Monitoring and Observability

Classic service monitoring is necessary but insufficient.

## Infra Metrics

- Latency
- Error rate
- Throughput
- Saturation
- GPU utilization
- Queue depth
- Cold starts
- Memory pressure

Tools: [[Prometheus]], Grafana, [[OpenTelemetry Protocol|OpenTelemetry]], Datadog, CloudWatch, etc.

## ML Metrics

- Data drift: feature distribution changed.
- Prediction drift: output distribution changed.
- Concept drift: relationship between input and target changed.
- Model performance: accuracy, precision/recall, calibration, business KPI.
- Segment/slice performance: model is fine overall but broken for a subgroup.
- Feature freshness: online features are stale.
- Label delay: true outcomes not available yet.

Tools: Evidently, WhyLabs/whylogs, Arize/Phoenix, Fiddler, custom warehouse dashboards.

Important: drift is a smoke alarm, not a diagnosis. Sometimes drift is harmless; sometimes no statistical drift appears and the product is still broken.


# Evals

Evals are the test suite for ML systems.

- Unit-style evals: known failure cases, cheap assertions.
- Golden datasets: curated examples that represent the product's actual task.
- Slice evals: performance by segment, language, customer type, geography, input length, etc.
- Regression evals: candidate model/prompt must not break known-good behavior.
- Human evals: necessary when the task is subjective or high-stakes.
- Online evals: production outcomes, feedback, business metrics.

For LLMs:
- Exact-match tests are rarely enough.
- LLM-as-judge can help, but needs calibration and spot-checking.
- Evaluate retrieval, tool use, structured output validity, factuality, refusal behavior, latency, and cost separately.

The meta-rule: ==evals should mirror the product behavior you actually care about.== Generic benchmark scores are trivia unless they predict your use case.


# LLMOps

[[LLMOps]] is MLOps with a different center of gravity.

For many LLM products, you are not training a model. The production artifact is a bundle:

```
prompt/version + model/provider + retrieval corpus + embedding model + tool graph + eval set + guardrails
```

That shifts the stack:
- Prompt management replaces some model registry use cases.
- Tracing becomes central because a request may involve retrieval, reranking, tool calls, multiple model calls, and post-processing.
- Evals become more important than training pipelines.
- Cost and latency are first-class product metrics.
- Data flywheel means collecting production traces, user feedback, corrections, labels, and hard negatives.

Tools:
- [[Langfuse]]: open-source LLM observability/evals/prompt management.
- [[LangSmith]]: LangChain ecosystem tracing/evals/datasets.
- [[Weave]]: W&B's LLM observability and eval layer.
- Arize Phoenix: open-source AI observability and tracing.
- Braintrust / Humanloop / Helicone / Portkey: adjacent eval, prompt, gateway, and observability tools.

Opinionated take from the LLM era: ==the model is not the product; the eval/data/feedback loop is the moat.==


# CI/CD for ML

ML CI/CD should test more than code.

Pull request checks:
- Unit tests
- Type/lint checks
- Data schema checks
- Feature validation
- Small training smoke test
- Eval suite on a fixed dataset
- Container build

Promotion checks:
- Full training run completed
- Model registered
- Eval report passed thresholds
- Slice metrics did not regress
- Security/license checks passed
- Human approval if needed
- Canary/shadow plan attached

Common tooling:
- [[GitHub Actions]], GitLab CI, Buildkite
- Docker/OCI images
- ArgoCD/Flux for GitOps on Kubernetes
- Terraform/Pulumi for infra
- MLflow/W&B/SageMaker/Vertex/Databricks APIs for promotion


# Governance and Safety

This becomes load-bearing in regulated or customer-facing ML.

- Access control for data, features, models, prompts, and traces.
- PII handling and redaction.
- Audit logs: who trained, approved, deployed, and rolled back.
- Model cards and dataset documentation.
- Bias/fairness checks where relevant.
- License tracking for datasets, model weights, and generated data.
- Secrets management.
- Supply-chain security for containers and dependencies.
- Retention policies for prompts/traces, especially if they contain user data.

LLM-specific:
- Prompt injection defenses.
- Tool permissioning.
- Output filtering/moderation.
- Human-in-the-loop for high-risk actions.
- Rate limits and spend caps.
- Retrieval corpus access control.


# Small-Team Defaults

## Classical ML

```
S3/Parquet + DuckDB/Polars/dbt
MLflow or W&B
GitHub Actions
FastAPI or BentoML
Evidently + Prometheus/Grafana
```

Avoid Kubeflow unless you already have a platform team.

## Enterprise ML Platform

```
Lakehouse + Catalog
Airflow/Dagster/Prefect
Feast/Tecton if online features matter
MLflow/W&B
Kubernetes + KServe/BentoML/Triton
OpenTelemetry + Evidently/Arize/Fiddler
GitOps + approvals
```

## LLM Product

```
Managed model API
Prompt versioning
RAG index pipeline
Langfuse/LangSmith/Weave
Golden eval set
Trace collection + human feedback
Gateway/caching/rate limits
Canary prompt/model releases
```

No GPUs before PMF unless privacy, latency, or unit economics force the issue.


# Trends to Internalize

1. ==Evals are the new tests.== Without evals, model changes are vibes.
2. ==Batch first.== Online inference is expensive complexity unless the product needs it.
3. ==Feature stores are not mandatory.== They solve real skew/reuse problems, but many teams do not have those problems yet.
4. ==Kubeflow is not the default.== It is for organizations that already operate Kubernetes as a platform.
5. ==MLflow and W&B are the center of gravity.== Most teams need tracking + artifacts + registry before they need exotic infrastructure.
6. ==LLMOps moved attention from training to evaluation, tracing, and feedback loops.==
7. ==Use managed models first.== Self-hosting LLMs is an infra commitment, not a personality trait.
8. ==The hard part is not training a model.== The hard part is knowing when it is safe and useful in production.


# Vocabulary

- [[Training-Serving Skew]]: mismatch between features/data used at training time and serving time.
- [[Data Drift]]: input distribution changes.
- [[Concept Drift]]: relationship between inputs and targets changes.
- [[Model Registry]]: versioned, governed repository for model artifacts.
- [[Feature Store]]: system for consistent offline/online feature computation and serving.
- [[Model Card]]: documentation for intended use, metrics, limitations, and risks.
- [[Shadow Deployment]]: run candidate model on production traffic without using its output.
- [[Canary Deployment]]: send a small slice of production traffic to a candidate.
- [[Champion Challenger]]: production model competes against candidate models.
- [[Data Flywheel]]: production usage generates feedback/labels that improve future versions.


# Mental Model

Modern MLOps is not one stack. It is a set of control loops:

```
Data quality loop
Experiment loop
Evaluation loop
Deployment loop
Monitoring loop
Feedback/retraining loop
```

The stack is good when those loops are short, observable, and boring.
