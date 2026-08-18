# Reflection

## Hardest part

The verification step, and specifically realising that my first design didn't work.

My first instinct was a mechanical ledger: pull every number out of the tool outputs, then filter out any number in the answer that isn't in that set. It's cheap, deterministic, and testable. But It fails on the failure mode that actually matters. If the agent looks at a `segment_stats` result showing DSL churning at 42% and writes "this customer's
risk score is 42%", the ledger passes it, because 42% really is in the tool output. The number is
real and the label on it is wrong, and no amount of value-matching can tell the difference.

A solution was to make LLM a judge, but it leads to a loop where an LLM evluates another LLM, can still hallucinate. That felt like circular reasoning until I read how RAGAS does faithfulness scoring, and the framing that unlocked it was that the judge is doing an easier job than the agent: it isn't answering the question, it's checking entailment between one claim and one structured tool result. That's a narrower task with the evidence already in front of it.

The third challenge was scope. My first version showed the judge the whole thread, which grows without bound and makes collisions between unrelated numbers more likely. Scoping the judge model to check from the Current-turn-only tool calls data was the obvious fix but was too strict, Anything referred by the model from the recent context, one turn later would force a pointless recompute. 

The fix for this was to have a sliding window of recent tool calls context from which to evaluate the faithfullnes

## What I learned

- Machine learning clarity of why and when to choose between ROC-AUC and PR-AUC
- Faithfullness check methods for data sensitive agentic apps
- About data anlysis and EDA queries (didnt have much experience on that side first)



## What I'd do differently with more time

- **Judge the judge.** The faithfulness judge is a single model call with no calibration. I'd build
a small labelled set of draft/tool-output pairs, some grounded, some mislabeled, some
fabricated, and measure its precision and recall, so its flag rate means something instead of
being an unquantified hope. 
- **Split the judge from the agent model.** They share `OPENROUTER_MODEL` today. A model checking
its own output shares its blind spots, and a cheap second opinion from a different family would
be a genuinely independent check for very little cost.
- **Cache the summaries properly.** Summaries are memoised in a process-local dict keyed by message  
id. It works for a single-user local app and would be wrong the moment there were two workers



## Time log

Stage 1 (notebook, model, evaluation) was done first and separately.


|                                                                            | hours   |
| -------------------------------------------------------------------------- | ------- |
| EDA, cleaning, model comparison, calibration, evaluation, notebook writeup | 3       |
| Planning agentic app, tool granularity, verification design, phase plan    | 1       |
| Model + dataset tool layers, factor enrichment, sandbox, tests             | 2       |
| Tool registry, faithfulness judge, agent graph, verified-turn wrapper      | 2       |
| FastAPI routes and SSE streaming                                           | 0.5     |
| React frontend                                                             | 1       |
| Docker, eval set, documentation                                            | 0.5     |
| Bug fixes                                                                  | 2       |
| **Total**                                                                  | **~12** |




## Where I stopped

- The faithfulness judge's own accuracy is unmeasured.
- No auth, no multi-user support, and the SQLite files are local, this is a single-user local app
and nothing about it is deployment-hardened.
- No prompt engineering techniques like Reflection, Tree of thought or self consistency
- Try a deep learning approach for the churn model

