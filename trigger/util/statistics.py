def to_range(percentage: float, step: int) -> str:
    lower = (int(percentage * 100) // step) * step
    upper = lower + step
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


def min_from_distribution(distribution) -> float:
    return min(int(number)
               for number
               in distribution.keys())
