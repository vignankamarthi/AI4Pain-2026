"""Integration: coevolutionary critic vs program population.

FRAMEWORK.md Section 5.2: Hillis-style arms race. Critic population evolves
to maximize program failure rates, program population evolves to handle
critic-induced perturbations. This test asserts:
  - critic.evolve_one returns a CriticIndividual
  - critic.hardest_critics top-N are surfaced for the next eval cycle
"""
import pytest
from framework import breakdown


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_critic_evolution_round():
    pop = breakdown.CriticPopulation(size=20)
    fake_failures = {"program_run_id_1": 0.3, "program_run_id_2": 0.1}
    crit = pop.evolve_one(program_population_failures=fake_failures)
    assert isinstance(crit, breakdown.CriticIndividual)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="TDD red")
def test_hardest_critics_surfaced_after_evolution():
    pop = breakdown.CriticPopulation(size=20)
    for _ in range(5):
        pop.evolve_one(program_population_failures={})
    top = pop.hardest_critics(n=3)
    assert len(top) <= 3
