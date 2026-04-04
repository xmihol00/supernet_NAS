from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from imx500_supernet import SubnetConfig

from .baseline_sga import SearchSpace


def _fitness_value(record: Dict[str, object]) -> float:
    value = record.get("fitness", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _birth_id_value(record: Dict[str, object]) -> int:
    value = record.get("birth_id", 0)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


@dataclass
class RegularizedEvolution:
    sample_size: int = 8
    mutation_rate: float = 0.35

    def _mutate_value(self, value: int, candidates: Sequence[int], rng: random.Random) -> int:
        if rng.random() >= self.mutation_rate:
            return value
        options = [candidate for candidate in candidates if candidate != value]
        if not options:
            return value
        return int(rng.choice(options))

    def mutate(self, config: SubnetConfig, search_space: SearchSpace, rng: random.Random) -> SubnetConfig:
        stage_depths = tuple(
            self._mutate_value(config.stage_depths[idx], search_space.stage_depth_candidates[idx], rng)
            for idx in range(4)
        )
        stage_widths = tuple(
            self._mutate_value(config.stage_widths[idx], search_space.stage_width_candidates[idx], rng)
            for idx in range(4)
        )
        return SubnetConfig(
            resolution=self._mutate_value(config.resolution, search_space.resolution_candidates, rng),
            stem_width=self._mutate_value(config.stem_width, search_space.stem_width_candidates, rng),
            stage_depths=stage_depths,
            stage_widths=stage_widths,
        )

    def propose(
        self,
        population: List[Dict[str, object]],
        search_space: SearchSpace,
        num_offspring: int,
        rng: random.Random,
    ) -> List[SubnetConfig]:
        if not population:
            return []

        offspring: List[SubnetConfig] = []
        for _ in range(num_offspring):
            sample = rng.sample(population, k=min(self.sample_size, len(population)))
            sample.sort(key=_fitness_value, reverse=True)
            parent = sample[0]["config"]
            child = self.mutate(parent, search_space, rng)
            offspring.append(child)
        return offspring

    def select_next_population(
        self,
        population: List[Dict[str, object]],
        offspring: List[Dict[str, object]],
        population_size: int,
    ) -> List[Dict[str, object]]:
        merged = [*population, *offspring]
        merged.sort(key=_birth_id_value)
        if len(merged) <= population_size:
            return merged
        return merged[-population_size:]
