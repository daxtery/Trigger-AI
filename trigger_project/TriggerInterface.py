from trigger_project.scoring_options import ScoringOptions
from .metrics.match import calculate_scores
from .models.trigger_match import TriggerMatch
from typing import Final, List
from .operation import Operation, OperationType
from trigger.train.cluster.Processor import Processor
from .models.user import User
from .instances.user_instance import UserInstance
from .transformers.user_transformer import UserTransformer
from .transformers.opening_transformer import OpeningTransformer
from .scoring_result import ScoringResult


class TriggerInterface:
    def __init__(self,
                 processor: Processor,
                 user_transformer: UserTransformer,
                 opening_transformer: OpeningTransformer,
                 scoring_options: ScoringOptions) -> None:
        self.processor: Final[Processor] = processor
        self.user_transfomer: Final[UserTransformer] = user_transformer
        self.opening_transformer: Final[OpeningTransformer] = opening_transformer
        self.scoring_options: Final[ScoringOptions] = scoring_options

    def on_operation(self, operation: Operation):

        if operation.type == OperationType.NEW_OPENING:
            opening_instance = operation.opening_instance
            opening = opening_instance.opening
            tag = operation.opening_instance_tag

            self.processor.process(tag, opening_instance.embedding, opening)

        elif operation.type == OperationType.REMOVE_OPENING:
            tag = operation.opening_instance_tag

            self.processor.remove(tag)

        elif operation.type == OperationType.UPDATE_OPENING:
            opening_instance = operation.opening_instance
            opening = opening_instance.opening
            tag = operation.opening_instance_tag

            self.processor.update(tag, opening_instance.embedding, opening)

    def calculate_matches_of_user(self, user: User) -> List[TriggerMatch]:
        user_instance = UserInstance(user, self.sentence_embedder)

        would_be_cluster_id = self.processor.predict(user_instance.embedding)

        instances, tags = self.processor.get_instances_and_tags_in_cluster(would_be_cluster_id)

        openings = [self.processor.get_custom_data_by_tag(tag) for tag in tags]

        scores = [
            calculate_scores(user_instance, opening, instance)
            for opening, instance in zip(openings, instances)
        ]

        matches = [
            TriggerMatch(user, score.final_score, score.opening)
            for score in scores
            if score.final_score >= 0.5
        ]

        return matches

    def compute_user_score_by_opening_tag(self, user: User, opening_id: str) -> ScoringResult:
        user_instance = UserInstance(user, self.sentence_embedder)

        opening = self.processor.get_custom_data_by_tag(opening_id)
        embedding = self.processor.get_instance_by_tag(opening_id)

        return calculate_scores(self.scoring_options, user, user_instance.embedding, opening, embedding)