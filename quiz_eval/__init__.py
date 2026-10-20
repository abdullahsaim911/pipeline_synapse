"""
Model-free tooling for the human review of Accessible Quiz & Practice question quality.

Nothing in this package calls a model, a provider, the quiz endpoint or any network service, and it
imports nothing from llm_interface/. It validates a dataset manifest and a reviewer scorecard,
and computes the contract's provisional release-gate metrics ONLY from items a human has reviewed.

Human judgments and automatically computed checks are kept apart: a person decides whether a
question is answerable or its key correct; the tool only counts those decisions and checks the
mechanical facts it can (does a quote occur in the transcript, is a time range inside the lecture).

Missing is not failing: an unjudged item is reported as "pending", a check that could not run (no
transcript available) as "unknown", and neither is counted as a failure.

Nothing here is evidence about the real model until a person has reviewed real output. Synthetic
data (is_synthetic: true) is accepted by the tools for testing and is never turned into a verdict.
See quiz_eval/README.md and quiz_eval/REVIEWER_GUIDE.md.
"""
