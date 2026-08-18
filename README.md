# Autonomous Churn Data Analyst

A churn model plus an agent that answers questions about the customer base by calling real tools
against the real data, and checks its own answer before it ships it.

---

## Running it

```bash
cp .env.example .env    # then put an OpenRouter key in OPENROUTER_API_KEY
```

```bash
docker compose up --build
```

Frontend on http://localhost:5173, backend on http://localhost:8000.

Without Docker:

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/uvicorn app.main:app --reload
```

```bash
cd frontend && npm install && npm run dev
```

Tests:

```bash
cd backend && .venv/bin/pytest
```

`OPENROUTER_MODEL` is a config value, not a hardcoded string point it at any OpenRouter model
slug. The same model is used for the agent, the summarizer and the judge.

---

## Stage 1 model

### Data issues found and how they were handled

1. `TotalCharges` stored as string; 11 blank-string rows, all `tenure==0` brand-new signups -> changed to numeric, imputed as `0.0`.
2. Six add-on columns' `"No internet service"` category is 100% redundant with `InternetService=='No'` (same for `MultipleLines`/`PhoneService`) -> collapsed to `"No"`.
3. `TotalCharges ≈ tenure × MonthlyCharges` (corr 0.83), this is a derived column and can be redundant
4. `gender` carries no real signal (26.9% vs 26.2% churn). So it is kept for transparency, not predictive value.

### Why PR-AUC over ROC-AUC

PR-AUC (0.636 score) as the evaluation metric, because of the class imbalance, precision is more important and PR-AUC is more relevant to a problem that's fundamentally about finding a minority class. 

The honest comparison is the multiple over each metric's own floor: the chosen model is about
**1.7× its ROC-AUC floor** but about **2.5× its PR-AUC floor**. The metric with the smaller
headline number is the one where the model is doing more real work.

### Why plain LogisticRegression, not `class_weight='balanced'`

| model | ROC-AUC | PR-AUC | Brier |
| --- | --- | --- | --- |
| LogReg (plain) | 0.8456 | 0.6584 | **0.1351** |
| LogReg (balanced) | 0.8454 | 0.6568 | 0.1653 |
| RandomForest | 0.8442 | 0.6528 | 0.1492 |
| HistGB | 0.8390 | 0.6468 | 0.1382 |

Balancing buys nothing on either ranking metric. it makes the predicted
probabilities meaningfully less trustworthy for no ranking benefit. Since every tool in this app
returns a *continuous* risk score that a human reads as a probability, calibration is the property
that actually matters here, so the plain model wins. 

The tree ensembles don't beat the linear
model on any of the three metrics, and the linear model gives exact per-feature contributions for
`top_factors` with no extra tooling.

---

## Stage 2 how the agent plans and self-checks

### Tool design

I have chosen 8 tools. Each tool is a small typed surface that generalises across many questions:

| tool | what it covers |
| --- | --- |
| `predict_churn_risk` | score one existing customer, with explained factors |
| `predict_hypothetical` | score a partial customer description, reporting defaulted fields |
| `compare_scenarios` | hypothetical customer or data points modification for existing; the delta is computed in Python, never by the model subtracting |
| `describe_column` | univariate stats / value counts / IQR outliers |
| `segment_stats` | filter + group by up to 2 dims + arbitrary aggregations |
| `rank_customers` | top-N on any column, including `risk_score` |
| `correlation` | dtype-dispatched association test |
| `run_pandas` | last-resort to run custom queries, have checks & guardrails |

Two design choices inside these are worth calling out:

**The delta is computed in Python.** `compare_scenarios` returns `baseline`, `modified` and
`delta` together. If the agent had to call the scorer twice and subtract, that subtraction would
be an unverifiable model-generated number sitting in the middle of the answer.

**Factor explanations are enriched structurally, not as a bolt-on.** `predict_churn_risk` calls
into the dataset layer for every factor it returns, so a factor arrives already carrying its
distributional context:

```json
{"feature": "tenure", "direction": "increases_risk", "contribution": 1.5479,
 "customer_value": 2,
 "context": {"percentile": 8.9, "decile_range": "0-2 months", "decile_n": 862,
             "decile_churn_rate": 0.5835, "baseline_churn_rate": 0.2654}}
```

The agent can therefore say "their 2-month tenure churns at 58% against a 27% baseline" without a second tool round-trip, and without inventing the
comparison. Mapping the model's transformed feature names (`InternetService_Fiber optic`) back to raw column and value is done by walking the fitted `OneHotEncoder`'s own categories, not by splitting the name on an underscore

### The self-check: 

**1. A single-call LLM faithfulness judge**, modelled on RAGAS's faithfulness metric. Instead of
asking "does this number exist", the judge is given the draft answer plus the structured tool
outputs and asked, per claim, whether the claim *including what it says it measures* is entailed
by a specific tool result. That is a semantic check, so mislabeling is exactly what it catches.
Claims come back classified `grounded` / `mislabeled` / `fabricated`. 

**2. Scoping the check to a turns window.** The judge checks the draft against the tool calls
inside a sliding window of the last `RECENT_TURNS_WINDOW` turns (default 3), not the whole thread
and not the current turn alone. Whole-thread grows context without bound. Current-turn-only is too
strict a natural follow-up one turn later ("what about DSL instead?") would force a wasteful
recompute of something still sitting in context. 

**What happens on a failure.** `run_verified_turn` re-invokes the agent once with a corrective
message naming the specific flagged claims and the judge's reasons. If the rewrite still fails,
the answer ships with a visible caveat rather than being silently withheld or silently shipped.

**Known limitation, stated plainly:** the judge is itself a model call. It can be lenient,
inconsistent between runs, or wrong in either direction, and it will be least reliable on exactly
the flaky free-tier models this is cheapest to run against. This substantially narrows the
hallucination-shaped failure surface. It is not a proof of correctness, and nothing here should be
read as one.

### Other guardrails

- The system prompt carries a full schema summary. every column, its dtype, its category levels,
  the row count and the baseline churn rate, as static text rather than a tool round-trip.
- The agent is told to refuse questions about fields the dataset doesn't have (region, revenue
  trend, product category) rather than answering against the nearest similar column. 
- Every tool returns `{"error": ...}` with the valid column names or category values on bad input.
  No stack trace reaches the UI; a failed turn becomes an `error` SSE event with a plain message.

---

## Eval set

`backend/tests/eval_set.py` runs 16 questions with known-correct answers through the full agent
univariate EDA, cohort aggregation, both correlation branches, top-N on a raw column, single
customer risk with enrichment context, a what-if, a hypothetical, a multi-step question needing
chained tools, an in-window follow-up, a trap question naming a column that does not exist, and an
invalid customer ID. It records the answer, the tools called, the judge's verdict and pass/fail,
and writes `docs/eval_report.md` with accuracy, faithfulness-flag rate and tool calls per question.

```bash
cd backend && .venv/bin/python -m tests.eval_set
```

## AI tool use

Gemini was used to understand the assignment thoroughly abd  Claude (Claude Code) was used to draft the notebook's narrative sections, to write
most of the application code from a phase-by-phase plan I wrote first, and to draft this README.
The design decisions tool granularity, the verification approach and the rejected
alternatives, the shared turns window, no decision threshold in the app  were mine and were
settled before any code was written. Everything was reviewed and corrected as it landed; the
model-layer numbers in this README and in the notebook were checked against the real artifacts.
