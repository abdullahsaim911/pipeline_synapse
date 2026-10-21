"""
Accessible Quiz & Practice: model-free backend foundations (Stage 1, PROPOSED).

Nothing in this package calls a model, defines an endpoint, or is wired into the
application. See docs/QUIZ_CONTRACT.md in the front-end repository.

Modules:
    limits     - provisional, configurable limits
    schemas    - strict request/response/question schemas (contract_version 1)
    passages   - transcript evidence passages and server-side evidence->time mapping
    screening  - deterministic, reject-only screening of untrusted candidates

Automatic screening is a rejection filter, NOT proof that a question or its answer
key is correct.
"""
