from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from imx500_supernet import SubnetConfig


def _fitness_value(record: Dict[str, object]) -> float:
    value = record.get("fitness", 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


@dataclass(frozen=True)
class SearchSpace:
    resolution_candidates: Sequence[int]
    stem_width_candidates: Sequence[int]
    stage_depth_candidates: Sequence[Sequence[int]]
    stage_width_candidates: Sequence[Sequence[int]]


class SimpleGeneticAlgorithm:
    def __init__(
        self,
        mutation_rate: float = 0.25,
        tournament_size: int = 3,
    ) -> None:
        self.mutation_rate = mutation_rate
        self.tournament_size = max(2, tournament_size)

    def _sample_parent(self, population: List[Dict[str, object]], rng: random.Random) -> Dict[str, object]:
        pool = rng.sample(population, k=min(self.tournament_size, len(population)))
        pool.sort(key=_fitness_value, reverse=True)
        return pool[0]

    def _mutate_value(self, value: int, candidates: Sequence[int], rng: random.Random) -> int:
        if rng.random() >= self.mutation_rate:
            return value
        options = [candidate for candidate in candidates if candidate != value]
        if not options:
            return value
        return int(rng.choice(options))

    def crossover(self, config_a: SubnetConfig, config_b: SubnetConfig, rng: random.Random) -> SubnetConfig:
        stage_depths = tuple(
            config_a.stage_depths[idx] if rng.random() < 0.5 else config_b.stage_depths[idx]
            for idx in range(4)
        )
        stage_widths = tuple(
            config_a.stage_widths[idx] if rng.random() < 0.5 else config_b.stage_widths[idx]
            for idx in range(4)
        )
        return SubnetConfig(
            resolution=config_a.resolution if rng.random() < 0.5 else config_b.resolution,
            stem_width=config_a.stem_width if rng.random() < 0.5 else config_b.stem_width,
            stage_depths=stage_depths,
            stage_widths=stage_widths,
        )

    def mutate(self, config: SubnetConfig, search_space: SearchSpace, rng: random.Random) -> SubnetConfig:
        stage_depths = tuple(
            self._mutate_value(
                value=config.stage_depths[idx],
                candidates=search_space.stage_depth_candidates[idx],
                rng=rng,
            )
            for idx in range(4)
        )
        stage_widths = tuple(
            self._mutate_value(
                value=config.stage_widths[idx],
                candidates=search_space.stage_width_candidates[idx],
                rng=rng,
            )
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
        offspring: List[SubnetConfig] = []
        if not population:
            return offspring

        for _ in range(num_offspring):
            parent_a = self._sample_parent(population, rng)
            parent_b = self._sample_parent(population, rng)
            base_child = self.crossover(parent_a["config"], parent_b["config"], rng)
            child = self.mutate(base_child, search_space, rng)
            offspring.append(child)

        return offspring

    def select_next_population(
        self,
        population: List[Dict[str, object]],
        offspring: List[Dict[str, object]],
        population_size: int,
    ) -> List[Dict[str, object]]:
        merged = [*population, *offspring]
        merged.sort(key=_fitness_value, reverse=True)
        return merged[:population_size]
