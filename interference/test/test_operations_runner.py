from pathlib import Path
from interference.transformers.transformer_pipeline import IdentityPipeline, NumpyToInstancePipeline, TransformerPipeline
from interference.clusters.processor import Processor
from interference.scoring import ScoringCalculator
from interference.util.json_encoder import EnhancedJSONEncoder
from interference.operations import Operation, OperationType

from interference.interface import Interface
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
        transformers: Dict[str, TransformerPipeline] = {
            "numpy": NumpyToInstancePipeline(),
            "identity": IdentityPipeline()
        },
        scoring_calculator: ScoringCalculator = ScoringCalculator(),
        only_output_evaluates: bool = True,
        output_base_folder: str = "",
        use_last_folder_name_as_processor_class: bool = True,
        output_type: str = 'json',
        skip_done: bool = False,
    ):
        self.processor_class = processor_class
        self.param_grid = param_grid
        self.operations = operations
        self.only_output_evaluates = only_output_evaluates
        
        if use_last_folder_name_as_processor_class:
            self.output_folder = os.path.join(output_base_folder, processor_class.__name__)
        else:
            self.output_folder = output_base_folder

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

    def init_inferface(self, params) -> Interface:
        processor = self.processor_class(**params)
        return Interface(processor, self.transformers, self.scoring_calculator)

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

    def run_test(self, interface: Interface):

        results = []

        for operation in self.operations:
            result = interface.on_operation(operation)
            if result is None:
                continue
            treated_result = self.after_operation_treat_result(interface, operation, result)
            if treated_result:
                results.append(treated_result)
        return results

    def after_operation_treat_result(self, interface: Interface, operation: Operation, result):

        if self.only_output_evaluates:
            if operation.type in [OperationType.EVALUATE_CLUSTERS, OperationType.EVALUATE_MATCHES]:
                return { "OperationType": operation.type, "Result": result }
            else:
                return None

        return { "Operation": operation.type, "Result": result }


    def _get_file_path(self, processor: Processor, output_type: str):
        file_name = processor.safe_file_name()
        return os.path.join(self.output_folder, F"{file_name}.{output_type}")

    def _save_results_json(self, file_path: str, interface: Interface, result, json_cls: Type[json.JSONEncoder] = None):

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
