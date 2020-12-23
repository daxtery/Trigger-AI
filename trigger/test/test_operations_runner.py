from pathlib import Path
from trigger.transformers.transformer_pipeline import TransformerPipeline
from trigger.clusters.processor import Processor
from trigger.scoring import ScoringCalculator
from trigger.util.json_encoder import EnhancedJSONEncoder
from trigger.operations import Operation

from trigger.trigger_interface import TriggerInterface
from typing import Any, Dict, List, Type

import logging

import json
import itertools
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_runner')
logger.setLevel(logging.INFO)


class TestRunner:

    def __init__(
        self,
        processor_class: Type[Processor],
        param_grid: Dict[str, Any],
        operations: List[Operation],
        transformers: Dict[str, TransformerPipeline] = {},
        scoring_calculator: ScoringCalculator = ScoringCalculator(),
        only_output_evaluates: bool = True,
        output_path: str = ".",
        output_type: str = 'json',
        skip_done: bool = False,
    ):
        self.processor_class = processor_class
        self.param_grid = param_grid
        self.operations = operations
        self.only_output_evaluates = only_output_evaluates
        self.output_path = output_path
        self.skip_done = skip_done
        self.transformers = transformers
        self.scoring_calculator = scoring_calculator

        if output_type != 'json' and output_type != 'csv':
            raise ValueError("Output file type not supported.")

        self.output_type = output_type
        self.tests = self._build_tests()

    def _build_tests(self):

        test_keys = self.param_grid.keys()

        test_values = self.param_grid.values()
        test_combinations = itertools.product(*test_values)

        test_items = [dict(zip(test_keys, test_item)) for test_item in test_combinations]

        return test_items

    def init_inferface(self, params) -> TriggerInterface:
        processor = self.processor_class(**params)
        return TriggerInterface(processor, self.transformers, self.scoring_calculator)

    def run_tests(self):

        for test in self.tests:

            interface = self.init_inferface(test)

            file_path = self._get_file_path(interface.processor, self.output_type)

            if self.skip_done and Path(file_path).exists():
                logger.info("Skipping test with params %s and output at %s. (file exists)", str(test), file_path)
                continue

            logger.info("Started test with params %s", str(test))

            results = self.run_test(interface)

            if self.output_type == 'json':

                self._save_results_json(file_path, interface, results, EnhancedJSONEncoder)

            else:

                self._save_results_csv(file_path, test, results)

    def run_test(self, interface: TriggerInterface):

        results = []

        for operation in self.operations:
            result = interface.on_operation(operation)

            if result:
                results.append({ "Operation": operation, "Result": result })
            elif not self.only_output_evaluates:
                results.append(operation)

        return results


    def _get_file_path(self, processor: Processor, output_type: str):
        file_name = processor.safe_file_name()
        return os.path.join(self.output_path, F"{file_name}.{output_type}")

    def _save_results_json(self, file_path: str, interface: TriggerInterface, result, json_cls: Type[json.JSONEncoder] = None):

        test_descriptor = {
            'algorithm': interface.processor.describe(),
            'interface': interface.describe(),
            'results': result
        }

        logger.info(f"Saving results to %s...", file_path)

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(test_descriptor, f, cls=json_cls)

    def _save_results_csv(self, file_path: str, params, result):
        pass
