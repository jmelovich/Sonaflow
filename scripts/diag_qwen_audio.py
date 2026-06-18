"""Diagnose whether the Qwen2-Audio processor actually encodes audio.

Builds inputs for two very different audio signals (loud sine vs silence) and
reports the processor output keys + whether the audio features differ. If the
features are identical/absent, the audio isn't reaching the model.
"""
from __future__ import annotations

import numpy as np
import transformers
from transformers import AutoProcessor

print("transformers", transformers.__version__)

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
proc = AutoProcessor.from_pretrained(MODEL)
sr = int(proc.feature_extractor.sampling_rate)
print("feature_extractor.sampling_rate", sr)

conv = [
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "x.wav"},
        {"type": "text", "text": "Describe this audio."},
    ]}
]
text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
print("---- templated text ----")
print(text)
print("------------------------")

t = np.linspace(0, 6.0, sr * 6, endpoint=False, dtype=np.float32)
sine = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
silence = np.zeros_like(sine)


def build(audio, label):
    try:
        inputs = proc(text=text, audio=[audio], sampling_rate=sr,
                      return_tensors="pt", padding=True)
    except Exception as exc:
        print(f"[{label}] audio= failed ({exc}); trying audios=")
        inputs = proc(text=text, audios=[audio], sampling_rate=sr,
                      return_tensors="pt", padding=True)
    keys = list(inputs.keys())
    print(f"[{label}] keys = {keys}")
    feat_key = next((k for k in ("input_features", "input_audio_features",
                                 "audio_values", "feature_attention_mask") if k in inputs), None)
    if feat_key is None:
        print(f"[{label}] NO AUDIO FEATURE KEY FOUND -> audio is being dropped!")
        return None
    feats = inputs.get("input_features", inputs.get(feat_key))
    print(f"[{label}] {feat_key} shape={tuple(feats.shape)} mean={feats.float().mean():.4f} std={feats.float().std():.4f}")
    return feats


fs = build(sine, "sine")
fz = build(silence, "silence")
if fs is not None and fz is not None and fs.shape == fz.shape:
    diff = (fs.float() - fz.float()).abs().mean().item()
    print(f"mean|sine-silence| = {diff:.5f}  ->", "DIFFER (good)" if diff > 1e-3 else "IDENTICAL (audio ignored!)")
