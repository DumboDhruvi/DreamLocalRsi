"""Candidate policy generation for offline replay dreaming."""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any, Dict, List, Optional

from dream_rsi.core.policy import ExplorationPolicy, PolicyConfig
from dream_rsi.policies.adaptive import AdaptivePolicy
from dream_rsi.policies.evolved import EvolvedPolicy
from dream_rsi.policies.fixed import FixedPolicy


class PolicyGenerator:
    """Generates candidate exploration policy configurations for replay simulation."""

    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)

    def generate_candidates(
        self,
        current_policy: ExplorationPolicy,
        num_candidates: int = 16,
    ) -> List[ExplorationPolicy]:
        """Generates candidate policies. Guarantees that current_policy is included as candidate 0 (monotonicity)."""
        candidates: List[ExplorationPolicy] = []

        # Candidate 0: Exactly the current policy
        candidates.append(current_policy)

        # Baseline FixedPolicy variant
        candidates.append(
            FixedPolicy(
                policy_id=f"fixed_cand_{uuid.uuid4().hex[:6]}",
                config=PolicyConfig(
                    branching_factor=max(1, current_policy.config.branching_factor),
                    parallel_workers=current_policy.config.parallel_workers,
                    stop_when_score=current_policy.config.stop_when_score,
                ),
            )
        )

        base_cfg = current_policy.config

        # Preset strategies
        # 1. High exploitation (greedily exploit high scores)
        candidates.append(
            EvolvedPolicy(
                policy_id=f"exploit_cand_{uuid.uuid4().hex[:6]}",
                config=PolicyConfig(
                    branching_factor=1,
                    prefer_high_score=0.75,
                    prefer_diversity=0.10,
                    prefer_unexplored=0.15,
                    exploration_constant=0.5,
                    parallel_workers=base_cfg.parallel_workers,
                    stop_when_score=base_cfg.stop_when_score,
                ),
                generation_metadata={"archetype": "high_exploitation"},
            )
        )

        # 2. High exploration (broad search across root & leaves)
        candidates.append(
            EvolvedPolicy(
                policy_id=f"explore_cand_{uuid.uuid4().hex[:6]}",
                config=PolicyConfig(
                    branching_factor=max(3, base_cfg.branching_factor + 1),
                    prefer_high_score=0.30,
                    prefer_diversity=0.30,
                    prefer_unexplored=0.40,
                    exploration_constant=2.5,
                    parallel_workers=base_cfg.parallel_workers,
                    stop_when_score=base_cfg.stop_when_score,
                ),
                generation_metadata={"archetype": "high_exploration"},
            )
        )

        # 3. High diversity (penalize similar files/strategies)
        candidates.append(
            EvolvedPolicy(
                policy_id=f"diversity_cand_{uuid.uuid4().hex[:6]}",
                config=PolicyConfig(
                    branching_factor=base_cfg.branching_factor,
                    prefer_high_score=0.40,
                    prefer_diversity=0.45,
                    prefer_unexplored=0.15,
                    diversity_weight=0.35,
                    parallel_workers=base_cfg.parallel_workers,
                    stop_when_score=base_cfg.stop_when_score,
                ),
                generation_metadata={"archetype": "high_diversity"},
            )
        )

        # 4. Fast early stop
        candidates.append(
            EvolvedPolicy(
                policy_id=f"early_stop_cand_{uuid.uuid4().hex[:6]}",
                config=PolicyConfig(
                    branching_factor=base_cfg.branching_factor,
                    prefer_high_score=0.50,
                    prefer_diversity=0.25,
                    prefer_unexplored=0.25,
                    stop_when_score=0.85,
                    parallel_workers=base_cfg.parallel_workers,
                ),
                generation_metadata={"archetype": "early_stop"},
            )
        )

        # Mutated variants
        while len(candidates) < num_candidates:
            mutated = self._mutate_config(base_cfg)
            candidates.append(
                EvolvedPolicy(
                    policy_id=f"mutant_{uuid.uuid4().hex[:6]}",
                    config=mutated,
                    generation_metadata={"archetype": "mutation"},
                )
            )

        return candidates[:num_candidates]

    def _mutate_config(self, base: PolicyConfig) -> PolicyConfig:
        cfg_dict = base.to_dict()

        # Mutate branching factor
        bf = cfg_dict.get("branching_factor", 2)
        bf = max(1, min(6, bf + self.rng.choice([-1, 0, 1])))

        # Mutate exploration constant
        c = cfg_dict.get("exploration_constant", 1.414)
        c = max(0.2, c * self.rng.uniform(0.7, 1.3))

        # Mutate diversity weight
        dw = cfg_dict.get("diversity_weight", 0.15)
        dw = max(0.0, min(0.6, dw + self.rng.uniform(-0.05, 0.05)))

        # Mutate weights
        w_score = max(0.1, cfg_dict.get("prefer_high_score", 0.5) + self.rng.uniform(-0.1, 0.1))
        w_div = max(0.1, cfg_dict.get("prefer_diversity", 0.3) + self.rng.uniform(-0.1, 0.1))
        w_unexp = max(0.1, cfg_dict.get("prefer_unexplored", 0.2) + self.rng.uniform(-0.1, 0.1))

        # Normalize preference weights to sum to 1.0
        total_w = w_score + w_div + w_unexp
        w_score = round(w_score / total_w, 3)
        w_div = round(w_div / total_w, 3)
        w_unexp = round(1.0 - w_score - w_div, 3)

        return PolicyConfig(
            branching_factor=bf,
            max_depth=base.max_depth,
            parallel_workers=base.parallel_workers,
            prefer_high_score=w_score,
            prefer_diversity=w_div,
            prefer_unexplored=w_unexp,
            stop_when_score=base.stop_when_score,
            exploration_constant=round(c, 3),
            diversity_weight=round(dw, 3),
        )
