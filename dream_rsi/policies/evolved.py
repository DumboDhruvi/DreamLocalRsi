"""Evolved policy parameterization tuned by offline dreaming."""

from __future__ import annotations

from typing import Any, Dict, Optional

from dream_rsi.core.policy import PolicyConfig
from dream_rsi.policies.adaptive import AdaptivePolicy


class EvolvedPolicy(AdaptivePolicy):
    """Exploration policy whose hyperparameters were optimized through offline replay dreaming."""

    def __init__(
        self,
        policy_id: str,
        config: Optional[PolicyConfig] = None,
        generation_metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(policy_id, config)
        self.generation_metadata = generation_metadata or {}
