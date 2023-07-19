from textattack.attack_recipes import AttackRecipe
from textattack.attack_results import SkippedAttackResult
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.transformations import WordSwap

from attack_utils import ICLUntargetedClassification, ICLInput, ICLClassificationGoalFunctionResult


class ICLTransferabilitySearch(SearchMethod):

    def perform_search(self, initial_result: ICLClassificationGoalFunctionResult):
        icl_input = initial_result.icl_input

        results, _ = self.get_goal_results([icl_input])
        result = results[0] if len(results) else None

        return result

    @property
    def is_black_box(self):
        return True


class ICLTransferabilityAttack(AttackRecipe):
    @staticmethod
    def build(model_wrapper):

        transformation = WordSwap()  # dummy transformation, not in use

        constraints = []

        #
        # Goal is ICL untargeted classification
        #
        goal_function = ICLUntargetedClassification(model_wrapper, use_cache=False)

        search_method = ICLTransferabilitySearch()

        return ICLTransferabilityAttack(goal_function, constraints, transformation, search_method)

    def attack(self, icl_input: ICLInput, ground_truth_output):
        assert isinstance(
            ground_truth_output, (int, str)
        ), "`ground_truth_output` must either be `str` or `int`."

        self.goal_function.model.ignore_attacked_text = True
        goal_function_result, _ = self.goal_function.init_attack_example(
            icl_input, ground_truth_output
        )

        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            self.goal_function.model.ignore_attacked_text = False

            result = self._attack(goal_function_result)
            return result
