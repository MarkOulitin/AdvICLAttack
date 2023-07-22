import textattack
import torch
from textattack import Attacker
from textattack.attack_results import SkippedAttackResult, SuccessfulAttackResult, MaximizedAttackResult, \
    FailedAttackResult
import tqdm
from textattack.shared.utils import logger

from custom_loggers import AllExamplesCSVLogger


class ICLAttacker(Attacker):
    def __init__(self, experiment_name, attack, dataset, attack_args=None, transferability: bool = False):
        self.experiment_name = experiment_name
        self.transferability = transferability

        super().__init__(
            attack,
            dataset,
            attack_args
        )

    def _attack(self):
        """Internal method that carries out attack.

        No parallel processing is involved.
        """
        self.attack_log_manager.loggers.append(AllExamplesCSVLogger(filename=f"./log_{self.experiment_name}_full_format.csv"))

        # add custom csv file with file color format
        self.attack_log_manager.add_output_csv(f"./log_{self.experiment_name}_diff_format.csv", "file")

        if torch.cuda.is_available():
            self.attack.cuda_()

        if self._checkpoint:
            num_remaining_attacks = self._checkpoint.num_remaining_attacks
            worklist = self._checkpoint.worklist
            worklist_candidates = self._checkpoint.worklist_candidates
            logger.info(
                f"Recovered from checkpoint previously saved at {self._checkpoint.datetime}."
            )
        else:
            if self.attack_args.num_successful_examples:
                num_remaining_attacks = self.attack_args.num_successful_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_successful_examples,
                    self.attack_args.shuffle,
                )
            else:
                num_remaining_attacks = self.attack_args.num_examples
                # We make `worklist` deque (linked-list) for easy pop and append.
                # Candidates are other samples we can attack if we need more samples.
                worklist, worklist_candidates = self._get_worklist(
                    self.attack_args.num_examples_offset,
                    len(self.dataset),
                    self.attack_args.num_examples,
                    self.attack_args.shuffle,
                )

        if not self.attack_args.silent:
            print(self.attack, "\n")

        pbar = tqdm.tqdm(total=num_remaining_attacks, smoothing=0, dynamic_ncols=True)
        if self._checkpoint:
            num_results = self._checkpoint.results_count
            num_failures = self._checkpoint.num_failed_attacks
            num_skipped = self._checkpoint.num_skipped_attacks
            num_successes = self._checkpoint.num_successful_attacks
        else:
            num_results = 0
            num_failures = 0
            num_skipped = 0
            num_successes = 0

        sample_exhaustion_warned = False
        while worklist:
            idx = worklist.popleft()
            try:
                if not self.transferability:
                    icl_input, ground_truth_output = self.dataset[idx]
                else:
                    attacked_icl_input, original_icl_input, ground_truth_output = self.dataset[idx]
            except IndexError:
                continue

            try:
                if not self.transferability:
                    result = self.attack.attack(icl_input, ground_truth_output)
                else:
                    result = self.attack.attack(attacked_icl_input, original_icl_input, ground_truth_output)
            except Exception as e:
                raise e
            if (
                    isinstance(result, SkippedAttackResult) and self.attack_args.attack_n
            ) or (
                    not isinstance(result, SuccessfulAttackResult)
                    and self.attack_args.num_successful_examples
            ):
                if worklist_candidates:
                    next_sample = worklist_candidates.popleft()
                    worklist.append(next_sample)
                else:
                    if not sample_exhaustion_warned:
                        logger.warn("Ran out of samples to attack!")
                        sample_exhaustion_warned = True
            else:
                pbar.update(1)

            self.attack_log_manager.log_result(result)
            if not self.attack_args.disable_stdout and not self.attack_args.silent:
                print("\n")
            num_results += 1

            if isinstance(result, SkippedAttackResult):
                num_skipped += 1
            if isinstance(result, (SuccessfulAttackResult, MaximizedAttackResult)):
                num_successes += 1
            if isinstance(result, FailedAttackResult):
                num_failures += 1
            pbar.set_description(
                f"[Succeeded / Failed / Skipped / Total] {num_successes} / {num_failures} / {num_skipped} / {num_results}"
            )

            if (
                    self.attack_args.checkpoint_interval
                    and len(self.attack_log_manager.results)
                    % self.attack_args.checkpoint_interval
                    == 0
            ):
                new_checkpoint = textattack.shared.AttackCheckpoint(
                    self.attack_args,
                    self.attack_log_manager,
                    worklist,
                    worklist_candidates,
                )
                new_checkpoint.save()
                self.attack_log_manager.flush()

        pbar.close()
        print()
        # Enable summary stdout
        if not self.attack_args.silent and self.attack_args.disable_stdout:
            self.attack_log_manager.enable_stdout()

        if self.attack_args.enable_advance_metrics:
            self.attack_log_manager.enable_advance_metrics = True

        self.attack_log_manager.log_summary()
        self.attack_log_manager.flush()
        print()
