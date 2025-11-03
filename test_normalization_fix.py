"""Test script for normalization with PipelineSpace."""

import neps
from neps.normalization import _normalize_imported_config


class TestSpace(neps.PipelineSpace):
    x = neps.Float(0, 1)
    y = neps.Integer(0, 10)
    epochs = neps.Fidelity(neps.Integer(1, 10))


space = TestSpace()

# Config with correct SAMPLING__ and ENVIRONMENT__ keys, plus an extra invalid key
config = {
    "SAMPLING__Resolvable.x::float__0_1_False": 0.5,
    "SAMPLING__Resolvable.y::integer__0_10_False": 5,
    "ENVIRONMENT__epochs": 3,
    "extra_key": 999,  # This should be removed
}

print("Input config keys:", sorted(config.keys()))

normalized = _normalize_imported_config(space, config)

print("Normalized config keys:", sorted(normalized.keys()))
print("Extra key removed:", "extra_key" not in normalized)
print("\nAll expected keys present:")
print(
    "  - SAMPLING__Resolvable.x::float__0_1_False:",
    "SAMPLING__Resolvable.x::float__0_1_False" in normalized,
)
print(
    "  - SAMPLING__Resolvable.y::integer__0_10_False:",
    "SAMPLING__Resolvable.y::integer__0_10_False" in normalized,
)
print("  - ENVIRONMENT__epochs:", "ENVIRONMENT__epochs" in normalized)
