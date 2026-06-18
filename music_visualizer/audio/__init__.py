"""Realtime audio analysis backend (milestone 2).

Extracts low-latency per-block audio features with aubio and feeds them to the
generation loop, where a :class:`~music_visualizer.audio.reactive.ReactiveMapper`
maps them onto the existing Deforum feedback-motion parameters.
"""
