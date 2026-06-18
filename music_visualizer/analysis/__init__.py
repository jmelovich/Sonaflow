"""Audio-understanding -> prompt-injection components.

This package turns realtime audio into Stable Diffusion prompt text. The CLAP
analyzer scores short rolling audio windows against a vocabulary of visual/mood
phrases; the prompt builder smooths those scores and composes compel-weighted
tags onto an editable base prompt; the orchestrator runs the loop on its own
thread and publishes the result to the active prompt.
"""
