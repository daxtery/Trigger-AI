from collections import Counter
import statistics
from trigger.util.statistics import average_from_distribution, max_from_distribution, min_from_distribution, to_range
from trigger.operations import CalculateMatchesInfo
from typing import Any, List

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trigger.trigger_interface import TriggerInterface


def eval_matches(interface: "TriggerInterface",
                 values_to_match: List[CalculateMatchesInfo]):

    by_value = []

    num_matches_counter = Counter()
    num_potential_counter = Counter()

    matches_range_counter = Counter()

    avg_similarity_range_counter = Counter()
    avg_matches_range_counter = Counter()

    for value_to_match in values_to_match:

        scorings = interface.get_scorings_for(
            value_to_match.transformer_key,
            value_to_match.value
        )

        if scorings is None:
            continue

        scoring_scores = [
            scoring.similarity_score
            for scoring in scorings
        ]

        matches = list(filter(lambda scoring: scoring.is_match, scorings))

        match_scores = [
            match.similarity_score
            for match in matches
        ]

        these_results = {
            'value': value_to_match,
            '#matches': len(match_scores),
            '#potential': len(scorings),
            'avg similarities': statistics.mean(scoring_scores) if len(scorings) > 0 else 0,
            'avg matches': statistics.mean(match_scores) if len(match_scores) > 0 else 0,
            'matches': list(matches)
        }

        by_value.append(these_results)

        num_matches_counter.update([these_results['#matches']])
        num_potential_counter.update([these_results['#potential']])

        for match in matches:
            matches_range_counter.update([to_range(match.similarity_score, 5)])

        if len(match_scores) > 0:
            avg_matches_range = to_range(these_results['avg matches'], 5)
            avg_matches_range_counter = avg_matches_range_counter + \
                Counter([avg_matches_range])

        avg_similarity_range = to_range(these_results['avg similarities'], 5)
        avg_similarity_range_counter = avg_similarity_range_counter + \
            Counter([avg_similarity_range])

    matches_count_distribution = {
        score: count
        for score, count in num_matches_counter.most_common()
    }

    potential_count_distribution = {
        score: count
        for score, count in num_potential_counter.most_common()
    }

    return {
        "distribution #matches": matches_count_distribution,
        "distribution #potential": potential_count_distribution,
        "distribution avg similarity range": {range_: count
                                              for range_, count in avg_similarity_range_counter.most_common()},
        "distribution avg matches range": {range_: count for range_, count in avg_matches_range_counter.most_common()},
        "distribution matches range": {range_: count for range_, count in matches_range_counter.most_common()},
        "% at least 1 match": 1 - (num_matches_counter.get(0, 0) / sum(num_matches_counter.values())),
        "avg #matches": average_from_distribution(matches_count_distribution),
        "max #matches of a value": max_from_distribution(matches_count_distribution),
        "min #matches of a value": min_from_distribution(matches_count_distribution),
        "avg #potential": average_from_distribution(potential_count_distribution),
        "max #potential of a value": max_from_distribution(potential_count_distribution),
        "min #potential of a value": min_from_distribution(potential_count_distribution),
        "avg matches score": statistics.mean([value["avg matches"] for value in by_value]),
        "avg similarity score": statistics.mean([value["avg similarities"] for value in by_value]),
        "by_value": by_value
    }