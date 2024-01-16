#!/usr/bin/env python3

"""
Official evaluation script of ConditionalQA.

To run this script (python3):
  python evaluate.py --pred_file=PATH_TO_YOUR_FILE --ref_file=PATH_TO_REF

"""

import json
import itertools
import math
import collections
import string
import re
import argparse


def evaluate(qid2predictions, qid2references):
    """Compute evaluation metrics."""
    # qid2predictions = {d["id"]: d["answers"] for d in predictions}
    # qid2references = load_answers(reference_filename)
    qid2references = {d["id"]: d["answers"] for d in qid2references}

    (total_em, total_conditional_em, total_f1, total_conditional_f1) = (
        list(),
        list(),
        list(),
        list(),
    )
    yesno_questions = list()
    extractive_questions = list()
    conditional_questions = list()

    i = 0
    for _, qid in enumerate(qid2references.keys()):
        if qid not in qid2predictions:
            em, conditional_em, f1, conditional_f1 = 0.0, 0.0, 0.0, 0.0
        else:
            em, conditional_em, f1, conditional_f1 = compute_metrics(
                qid2predictions[qid], qid2references[qid]
            )

            total_em.append(em)
            total_conditional_em.append(conditional_em)
            total_f1.append(f1)
            total_conditional_f1.append(conditional_f1)

            if not qid2references[qid]:
                pass
            elif any(ans[0] in ["yes", "no"] for ans in qid2references[qid]):
                yesno_questions.append(i)
            else:
                extractive_questions.append(i)

            if any(ans[1] for ans in qid2references[qid]):
                conditional_questions.append(i)

            i += 1

    def update_metrics(questions, prefix=""):
        return {
            prefix + "EM": sum(total_em[i] for i in questions) / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix
            + "EM_with_conditions": sum(total_conditional_em[i] for i in questions)
            / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix + "F1": sum(total_f1[i] for i in questions) / len(questions)
            if len(questions) > 0
            else 0.0,
            prefix
            + "F1_with_conditions": sum(total_conditional_f1[i] for i in questions)
            / len(questions)
            if len(questions) > 0
            else 0.0,
        }

    return {
        "total": update_metrics(range(len(total_em))),
        "yesno": update_metrics(yesno_questions),
        "extractive": update_metrics(extractive_questions),
        "conditional": update_metrics(conditional_questions),
    }


def load_answers(filename):
    with open(filename) as f:
        data = json.load(f)
    id2answers = {d["id"]: d["answers"] for d in data}
    return id2answers


def compute_metrics(prediction, reference):
    """
    Compute metrics for one example.

    args:
      prediction: a list of tuples of predicted answers and
        conditions, e.g. [(ans1, [c1, c2]), (ans2, [c3])]
      reference: same as prediction

    returns:
      A tuple of scalars for (em, em_with_conditions,
        f1, and f1_with_conditions)
    """

    # get full scores only if no answer is predicted
    if not reference:
        return [float(not prediction)] * 4

    num_answer = len(reference)

    if len(prediction) < num_answer:
        prediction.extend([("", list())] * (num_answer - len(prediction)))

    # iterate through all possible permutations
    max_em, max_f1 = 0.0, 0.0
    max_conditional_em, max_conditional_f1 = 0.0, 0.0
    for ordered_prediction in itertools.permutations(prediction):
        total_em, total_f1 = 0.0, 0.0
        total_conditional_em, total_conditional_f1 = 0.0, 0.0
        # compute metrics for one pair of answers
        for pred_answer, ref_answer in zip(ordered_prediction, reference):
            em, conditional_em, f1, conditional_f1 = compute_em_f1(
                pred_answer, ref_answer
            )
            total_em += em
            total_conditional_em += conditional_em
            total_f1 += f1
            total_conditional_f1 += conditional_f1

        # record the best permutation
        max_em = max(max_em, total_em / num_answer)
        max_conditional_em = max(max_conditional_em, total_conditional_em / num_answer)
        max_f1 = max(max_f1, total_f1 / num_answer)
        max_conditional_f1 = max(max_conditional_f1, total_conditional_f1 / num_answer)

    assert max_em <= 1 and max_f1 <= 1
    assert max_conditional_em <= 1 and max_conditional_f1 <= 1

    # discounted by extra predicted answers
    gamma = math.exp(1.0 - len(prediction) / num_answer)
    max_em *= gamma
    max_f1 *= gamma
    max_conditional_em *= gamma
    max_conditional_f1 *= gamma

    return max_em, max_conditional_em, max_f1, max_conditional_f1


def compute_em_f1(pred_answer, ref_answer):
    """
    Compute EM, F1 and with conditions for one answer.

    args:
      pred_answer: a tuple of (answer, conditions)
      ref_answer: a tuple of (answer, conditions)

    returns:
      EM, F1, and EM and F1 with conditions
    """
    conditions_f1 = compute_conditions_f1(pred_answer[1], ref_answer[1])

    pred_answer_text = normalize_answer(pred_answer[0])
    ref_answer_text = normalize_answer(ref_answer[0])
    em = float(pred_answer_text == ref_answer_text)
    f1 = compute_answer_f1(ref_answer_text, pred_answer_text)

    conditional_em = em * conditions_f1
    conditions_f1 = f1 * conditions_f1
    return em, conditional_em, f1, conditions_f1


def compute_conditions_f1(predicted_conditions, true_conditions):
    """
    Compute F1 of the predicted set of conditions.

    args:
      predicted_conditions: a list of predicted conditions
      true_conditions: a list of true conditions

    returns:
      element-wise condition F1
    """
    if not true_conditions:
        return float(not predicted_conditions)

    if not predicted_conditions:
        return 0.0

    true_conditions = list(set(true_conditions))
    predicted_conditions = list(set(predicted_conditions))

    correct = sum([int(c in true_conditions) for c in predicted_conditions])
    precision = correct / len(predicted_conditions)
    recall = correct / len(true_conditions)

    if correct == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 / (1.0 / precision + 1.0 / recall)

    return f1


##############################################################
###################### Helper Functions ######################
##############################################################


def compute_answer_f1(a_gold, a_pred):
    """Copied from SQuAD 2.0 evaluation script."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_tokens(s):
    """Copied from SQuAD 2.0 evaluation script."""
    if not s:
        return []
    return normalize_answer(s).split()


def normalize_answer(s):
    """Copied from SQuAD 2.0 evaluation script."""
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parse_arguments():
    # command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        dest="pred_file",
        type=str,
        default=None,
        help="Path to your prediction file.",
    )
    parser.add_argument(
        "--ref_file",
        dest="ref_file",
        type=str,
        default=None,
        help="Path to the reference file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(evaluate(args.pred_file, args.ref_file))
