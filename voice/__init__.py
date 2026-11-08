"""
Voice control backend for Synapse.

Separate stages, each in its own module:
  recognition     speech -> text          (faster-whisper, CPU)
  interpretation  text + context -> action (multilingual embeddings)
  feedback        short spoken replies    (existing TTSEngine)
See docs/voice-control.md.
"""
