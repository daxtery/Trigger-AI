from collections import Counter
import statistics
from trigger.util.statistics import average_from_distribution, max_from_distribution, min_from_distribution, to_range
from trigger.operations import CalculateMatchesInfo
from typing import Any, List

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trigger.trigger_interface import TriggerInterface


def eval_matches(interface: "TriggerInterface",
                 values_to_match: List[CalculateMatchesInfo[Any]],
                 fetch_matched_value: bool = False,
                 ):

    by_value = []

    num_matches_counter = Counter()
    num_potential_counter = Counter()

    matches_range_counter = Counter()

    avg_similarity_range_counter = Counter()
    avg_matches_range_counter = Counter()

    for value_to_match in values_to_match:

        scorings = interface.get_scorings_for(
            [value_to_match.transformer_key],
            [value_to_match.value]
        )[0]

        scoring_scores = [
            scoring.similarity_score
            for scoring in scorings
        ]

        matches = filter(lambda scoring: scoring.is_match, scorings)

        match_scores = [
            match.similarity_score
            for match in matches
        ]

        if fetch_matched_value:
            save_matches = [
                (match, interface.get_instances_by_tag([match.with_tag])[0])
                for match in matches
            ]
        else:
            save_matches = list(matches)

        these_results = {
            'value': value_to_match,
            '#matches': len(match_scores),
            '#potential': len(scorings),
            'avg similarities': statistics.mean(scoring_scores) if len(scorings) > 0 else 0,
            'avg matches': statistics.mean(match_scores) if len(match_scores) > 0 else 0,
            'matches': save_matches
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
        "max #matches of a user": max_from_distribution(matches_count_distribution),
        "min #matches of a user": min_from_distribution(matches_count_distribution),
        "avg #potential": average_from_distribution(potential_count_distribution),
        "max #potential of a user": max_from_distribution(potential_count_distribution),
        "min #potential of a user": min_from_distribution(potential_count_distribution),
        "avg matches score": statistics.mean([user["avg matches"] for user in by_value]),
        "avg similarity score": statistics.mean([user["avg similarities"] for user in by_value]),
        "by_user": by_value
    }