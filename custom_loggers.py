"""
Attack Logs to CSV
========================
"""

import csv

import pandas as pd
from textattack.loggers import Logger
from textattack.shared import logger



class AllExamplesCSVLogger(Logger):

    def __init__(self, filename="results.csv"):
        logger.info(f"Logging to CSV at path {filename}")
        self.filename = filename
        self.row_list = []
        self._flushed = True

    def log_attack_result(self, result):
        #result.original_result.icl_input.construct_prompt()
        #result.perturbed_result.icl_input.construct_prompt() TODO
        #result.original_result.icl_input.example_sentences[result.original_result.icl_input.pertubation_example_sentence_index] = result.original_result.icl_input.attacked_text.text

        # we update the example sentences for icl attack, all exampels attack is already handled
        # result.perturbed_result.icl_input.example_sentences[result.perturbed_result.icl_input.pertubation_example_sentence_index] = result.perturbed_result.attacked_text.text

        # print("final")
        # print(result.original_result.icl_input.example_sentences)
        # print("##############")
        original_text = '!@icl_attack_seperator@!'.join(result.original_result.icl_input.example_sentences)
        perturbed_text = '!@icl_attack_seperator@!'.join(result.perturbed_result.icl_input.example_sentences)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
        }
        self.row_list.append(row)
        self._flushed = False

    def flush(self):
        self.df = pd.DataFrame.from_records(self.row_list)
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True

    def close(self):
        # self.fout.close()
        super().close()

    def __del__(self):
        if not self._flushed:
            logger.warning("CSVLogger exiting without calling flush().")
