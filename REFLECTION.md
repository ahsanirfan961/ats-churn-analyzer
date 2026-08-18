# Reflection

## Hardest part

The verification step, and specifically realising that my first design didn't work.

My first instinct was a mechanical ledger: pull every number out of the tool outputs, then refuse
to ship any number in the answer that isn't in that set. It's cheap, deterministic, and testable —
everything you want in a guardrail. It also fails on the failure mode that actually matters. If the
agent looks at a `segment_stats` result showing DSL churning at 42% and writes "this customer's
risk score is 42%", the ledger passes it, because 42% really is in the tool output. The number is
real and the label on it is wrong, and no amount of value-matching can tell the difference.

Accepting that meant accepting that the check had to be semantic, which meant putting an LLM inside
my own guardrail — a model call checking a model call. That felt like circular reasoning until I
read how RAGAS does faithfulness scoring, and the framing that unlocked it was that the judge is
doing an easier job than the agent: it isn't answering the question, it's checking entailment
between one claim and one structured tool result. That's a narrower task with the evidence already
in front of it.

The third iteration was scope. My first version showed the judge the whole thread, which grows
without bound and makes collisions between unrelated numbers more likely. Current-turn-only was the
obvious fix and was too strict — "what about DSL instead?" one turn later would force a pointless
recompute. Landing on a sliding turns window, and then noticing that this is *the same boundary*
the message summarizer needs, was the moment the design stopped fighting itself. One constant now
drives both, so what the agent can see and what the judge can verify can never drift apart.

## What I learned

Most of the agent stack was new to me. LangGraph's `create_react_agent` was easy to start with and
had a sharp edge: the version pinned first had no `pre_model_hook`, so my planned summarization
design simply didn't exist in that API. I checked the installed signature before writing against
it, found `state_modifier` instead, then upgraded and found that the prebuilt summarization node
only windows by *token* budget, not by turns. Since my judge scope is defined in turns, using a
token approximation would have let the two windows disagree — which is exactly the bug the shared
constant exists to prevent — so I wrote the turn split myself. It's about twenty lines and it's
exactly right, which was the better trade than bending my design to fit a library helper.

The other thing I'd underweighted is how much of an agent's behaviour lives in text that isn't
code. The pydantic `Field` descriptions and tool descriptions are prompt content the model reads
on every call; they did more for reliability than any control-flow change I made.

## What I'd do differently with more time

- **Judge the judge.** The faithfulness judge is a single model call with no calibration. I'd build
  a small labelled set of draft/tool-output pairs — some grounded, some mislabeled, some
  fabricated — and measure its precision and recall, so its flag rate means something instead of
  being an unquantified hope. Right now the eval set reports how often it fires, not how often it's
  right.
- **Split the judge from the agent model.** They share `OPENROUTER_MODEL` today. A model checking
  its own output shares its blind spots, and a cheap second opinion from a different family would
  be a genuinely independent check for very little cost.
- **Cache the summaries properly.** Summaries are memoised in a process-local dict keyed by message
  id. It works for a single-user local app and would be wrong the moment there were two workers.
- **Confidence intervals on segment stats.** `segment_stats` reports a churn rate for any cohort,
  including cohorts of 12 customers, and the agent will happily quote it to four decimal places. A
  sample size is returned alongside it, but nothing forces the agent to treat a small `n` as
  uncertain, and it should.

## Time log

Stage 1 (notebook, model, evaluation) was done first and separately.

| | hours |
| --- | --- |
| Stage 1 — EDA, cleaning, model comparison, calibration, evaluation, notebook writeup | 8 |
| Planning Stage 2/3 — tool granularity, verification design, phase plan | 2 |
| Model + dataset tool layers, factor enrichment, sandbox, tests | 3 |
| Tool registry, faithfulness judge, agent graph, verified-turn wrapper | 3.5 |
| FastAPI routes and SSE streaming | 1 |
| React frontend | 1.5 |
| Docker, eval set, documentation | 1.5 |
| **Total** | **~20.5** |

## Where I stopped

- The Docker images build from the Dockerfiles as written, but I did not get a full
  `docker compose up` run through the browser on this machine, so treat the container path as
  built-but-not-smoke-tested.
- The faithfulness judge's own accuracy is unmeasured (see above).
- No auth, no multi-user support, and the SQLite files are local — this is a single-user local app
  and nothing about it is deployment-hardened.
- The frontend renders assistant messages as plain text. Markdown tables from the agent will show
  as raw pipes.
