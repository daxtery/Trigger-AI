from collections import Counter
from typing import Any, Dict, Optional, Tuple


Min = float
Max = float
Avg = float
Distribution = Dict[Any, int]
Range = str


def to_range(percentage: float, step: int) -> Range:
    lower = (int(percentage * 100) // step) * step
    upper = min(lower + step, 100)
    return f"{lower} - {upper}"


def extract_first_number_from_range(range_: str) -> int:
    return [int(s) for s in range_.split() if s.isdigit()][0]


def average_from_distribution(distribution) -> float:
    return sum(int(number) * frequency
               for number, frequency
               in distribution.items()) / \
        sum(frequency
            for frequency
            in distribution.values())


def max_from_distribution(distribution) -> float:
    return max(int(number)
               for number
               in distribution.keys())


def min_from_distribution(distribution: Distribution) -> float:
    return min(int(number)
               for number
               in distribution.keys())


Stats = Tuple[Distribution, Optional[Tuple[Avg, Max, Min]]]


def stats_from_counter(counter: Counter) -> Stats:

    distribution = {
        score: count
        for score, count in counter.most_common()
    }

    first = counter.most_common()[0][0]

    if(isinstance(first, int)):
        max = max_from_distribution(distribution)
        min = min_from_distribution(distribution)
        avg = average_from_distribution(distribution)
        return distribution, (avg, max, min)

    return distribution, None
