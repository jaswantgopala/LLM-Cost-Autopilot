"""
Router: ties together complexity classification and the tier-to-model routing map.
"""

import os
import pickle
import yaml
import numpy as np

from core.complexity import extract_features, features_to_vector
from core.registry import get_model, ModelConfig

CLASSIFIER_PATH = "data/complexity_classifier.pkl"
ROUTING_CONFIG_PATH = "config/routing.yaml"


class Router:
    def __init__(self):
        self._classifier_bundle = self._load_classifier()
        self._routing_config = self._load_routing_config()

    def _load_classifier(self):
        with open(CLASSIFIER_PATH, "rb") as f:
            return pickle.load(f)

    def _load_routing_config(self):
        with open(ROUTING_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)["routing"]

    def predict_tier(self, prompt: str) -> int:
        """Predict complexity tier (1, 2, or 3) for a given prompt."""
        features = extract_features(prompt)
        vector = np.array([features_to_vector(features)], dtype=float)

        model = self._classifier_bundle["model"]
        scaler = self._classifier_bundle["scaler"]

        if scaler is not None:
            vector = scaler.transform(vector)

        tier = int(model.predict(vector)[0])
        return tier

    def get_model_for_tier(self, tier: int) -> ModelConfig:
        tier_key = f"tier_{tier}"
        model_key = self._routing_config[tier_key]["model_key"]
        return get_model(model_key)

    def get_fallback_model_for_tier(self, tier: int) -> ModelConfig:
        tier_key = f"tier_{tier}"
        fallback_key = self._routing_config[tier_key]["fallback_model_key"]
        return get_model(fallback_key)

    def route(self, prompt: str) -> dict:
        tier = self.predict_tier(prompt)
        model_config = self.get_model_for_tier(tier)

        return {
            "prompt": prompt,
            "predicted_tier": tier,
            "tier_name": self._routing_config[f"tier_{tier}"]["name"],
            "routed_model_key": self._get_model_key_for_tier(tier),
            "model_config": model_config,
        }

    def _get_model_key_for_tier(self, tier: int) -> str:
        return self._routing_config[f"tier_{tier}"]["model_key"]