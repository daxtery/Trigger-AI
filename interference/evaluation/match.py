from collections import Counter
import statistics
from interference.scoring import Scoring
from interference.transformers.transformer_pipeline import Instance
from interference.util.statistics import Stats, stats_from_counter, to_range
from typing import Dict, Sequence
import numpy as np

def bin_deviation_score(
        test_distribution: Dict[str, int],
        to_compare_distribution: Dict[str, int]
    ) -> float:

    def normed_interval(range: str, maxi: int, mini: int) -> float:
        upper_bound = eval(range.split('-')[-1].strip())
        return (upper_bound - mini)/(maxi - mini)
    
    test_ranges = set(test_distribution.keys())
    to_compare_ranges = set(to_compare_distribution.keys())

    all_ranges = test_ranges.union(to_compare_ranges)

    all_ranges_lower_bound = [int(range.split("-")[0].strip()) for range in all_ranges]
    all_ranges_upper_bound = [int(range.split("-")[1].strip()) for range in all_ranges]

    min_range_bound = min(all_ranges_lower_bound)
    max_range_bound = max(all_ranges_upper_bound)

    scores = []

    for range in all_ranges:

        test_range = test_distribution.get(range, 0)
        to_compare_range = to_compare_distribution.get(range, 0)

        diff = test_range - to_compare_range

        w_diff = diff * np.exp(normed_interval(range, max_range_bound, min_range_bound))

        scores.append(w_diff)

    return np.sum(scores)

def eval_matches(
        instances_to_match: Sequence[Instance],
        individual_scorings: Sequence[Sequence[Scoring]]
        ):

    by_instance = []

    num_matches_counter = Counter()
    num_potential_counter = Counter()

    matches_score_range_counter = Counter()
    score_range_counter = Counter()

    avg_matches_score_range_counter = Counter()
    avg_score_range_counter = Counter()

    for instance, scorings in zip(instances_to_match, individual_scorings):

        scoring_scores = [
            scoring.score
            for scoring in scorings
        ]

        matches = list(filter(lambda scoring: scoring.is_match, scorings))

        match_scores = [
            match.score
            for match in matches
        ]

        these_results = {
            'value': instance.value,
            '#matches': len(match_scores),
            '#potential': len(individual_scorings),
            'average score': statistics.mean(scoring_scores) if len(individual_scorings) > 0 else 0,
            'average match score': statistics.mean(match_scores) if len(match_scores) > 0 else 0,
            'matches': list(matches)
        }

        by_instance.append(these_results)

        num_matches_counter.update([these_results['#matches']])
        num_potential_counter.update([these_results['#potential']])

        for match in matches:
            matches_score_range_counter.update([to_range(match.similarity_score, 5)])

        for scoring in scorings:
            score_range_counter.update([to_range(scoring.similarity_score, 5)])

        if len(match_scores) > 0:
            avg_matches_range = to_range(these_results['average match score'], 5)
            avg_matches_score_range_counter += Counter([avg_matches_range])
            
        avg_score_range = to_range(these_results['average score'], 5)
        avg_score_range_counter += Counter([avg_score_range])


    json_obj = {}

    def add_stats_to_json(name: str, stats: Stats):
        nonlocal json_obj
        distribution, number_stats = stats
        json_obj[f"distribution {name}"] = distribution

        if (number_stats):
            avg, max, min = number_stats
            json_obj[f"average {name}"]= avg
            json_obj[f"max {name}"]= max
            json_obj[f"min {name}"]= min

    add_stats_to_json("#matches", stats_from_counter(num_matches_counter))
    add_stats_to_json("matches score range", stats_from_counter(matches_score_range_counter))
    add_stats_to_json("average matches score range", stats_from_counter(avg_matches_score_range_counter))

    json_obj["% at least 1 match"] = (1 - (num_matches_counter.get(0, 0) / sum(num_matches_counter.values()))) * 100

    add_stats_to_json("#potential", stats_from_counter(num_potential_counter))

    add_stats_to_json("score range", stats_from_counter(score_range_counter))
    add_stats_to_json("average score range", stats_from_counter(avg_score_range_counter))

    json_obj["by_instance"] = by_instance

    return json_obj