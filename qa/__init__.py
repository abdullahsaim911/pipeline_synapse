"""
Ask a Question (transcript-grounded lecture Q&A).

Modules:
    transcript  - load and normalize timestamped transcript segments
    retrieval   - keyword/time-based evidence selection (no embeddings)
    moment      - validated current-moment (intervention) context
    tokens      - local token estimates for the prompt budget
    service     - prompt building and answer generation via an LLM provider
"""
