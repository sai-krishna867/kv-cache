# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from https://github.com/NVIDIA/RULER at commit
# db43f92ff634bbce1c33d1fb15858aec4bba509a:
#   - string_match_part, string_match_all, TASKS: scripts/eval/synthetic/constants.py (verbatim)
#   - postprocess_pred: scripts/eval/evaluate.py (verbatim)
# Copied rather than imported because upstream evaluate.py imports NeMo.

import re

RULER_COMMIT = "db43f92ff634bbce1c33d1fb15858aec4bba509a"


def postprocess_pred(predict_str: str, task_config: dict):

    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def string_match_part(preds, refs):
    score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)

def string_match_all(preds, refs):
    score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


TASKS = {
    'niah': {
        'metric_fn': string_match_all,
    },
    'variable_tracking': {
        'metric_fn': string_match_all,
    },
    'common_words_extraction': {
        'metric_fn': string_match_all,
    },
    'freq_words_extraction': {
        'metric_fn': string_match_all
    },
    'qa': {
        'metric_fn': string_match_part,
    },
}
