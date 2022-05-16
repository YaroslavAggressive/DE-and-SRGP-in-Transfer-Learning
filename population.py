from dataclasses import dataclass, field
import numpy as np
from scipy.stats import truncnorm


@dataclass
class Population:

    size: int = 0
    dimensions: int = 0
    individuals: np.array = field(default_factory=list)

    def __post_init__(self):
        self.individuals = np.array([])

    def __getitem__(self, key) -> np.array:
        return self.individuals[key]

    def __setitem__(self, key, value):
        self.individuals[key] = value

    def __iter__(self):
        return iter(self.individuals)

    def __sizeof__(self):
        return len(self.individuals)

    @staticmethod
    def random_population(size: int, dimensions: int) -> "Population":
        population = Population(size, dimensions)
        population.individuals = np.array([truncnorm.rvs(a=0, b=1, size=dimensions) for i in range(size)])
        return population


