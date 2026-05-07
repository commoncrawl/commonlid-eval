from statistics import mean
from typing import Any, Dict, List, Optional, Tuple, Iterable

from datasets import Dataset

from tests.legacy.langid_models import EvalLangIDModels, conform_langcode_with_reason
from collections import Counter
from tqdm.auto import tqdm
import time
import pandas as pd

from iso639.iso639 import Lang
from iso639.exceptions import InvalidLanguageValue, DeprecatedLanguageValue


def filter_by_language_frequency_and_sample(
        df: pd.DataFrame,
        language_column: str,
        min_count,
        n_samples_per_language
):
    # Filter languages with at least min_count samples
    df = df.groupby(language_column).filter(lambda x: len(x) >= min_count)
    # Sample n_samples per language
    df = df.groupby(language_column).apply(lambda grp: grp.sample(n=n_samples_per_language)).reset_index(drop=True)
    return df



class EvalLangIDDatasets:
    def __init__(self):
        self.datasets = {}

    def register_dataset(self, label: str, dataset: Dataset, text_column: str, target_iso639_3_column: str):
        if label in self.datasets:
            print(f"** dataset with label {label} already registered -> overwriting with incoming dataset")
        _check_iso693_3_column(dataset, target_iso639_3_column, label)
        self.datasets[label] = {
            "dataset": dataset,
            'text_column': text_column,
            'target_iso639_3_column': target_iso639_3_column
        }

    def eval_all(self, models: EvalLangIDModels, batch_size=16, slow_models_to_skip: Optional[List[str]] = None, sample_count_threshold: int = 0) -> Tuple[List[dict], Dict[str, Counter]]:
        """
        Evaluate all registered datasets with all language identification models.
        Returns tuple of records (list of dicts containing language, model, dataset, and metrics values) and errors (dict of Counters, one per dataset).
        """
        if len(self.datasets) == 0:
            print("No datasets registered - nothing to do")
            return [], {}

        all_records = []
        all_errors = {}

        for ds_index, (ds_label, ds_info) in enumerate(self.datasets.items()):
            try:
                print(f"Evaluating dataset {ds_index+1}/{len(self.datasets)}: {ds_label}")
                this_dataset_results, errors = _eval_all_models_on_dataset(
                    models=models,
                    dataset_label=ds_label,
                    dataset=ds_info['dataset'],
                    text_column=ds_info['text_column'],
                    target_iso639_3_column=ds_info['target_iso639_3_column'],
                    slow_models_to_skip=slow_models_to_skip,
                    batch_size=batch_size,
                )
                records = _to_records(models, this_dataset_results, sample_count_threshold=sample_count_threshold)
                for r in records:
                    r['dataset'] = ds_label
                all_errors[ds_label] = errors
                all_records.extend(records)
            except KeyboardInterrupt:
                print("Interrupted by user, returning what we have so far.")
                break

        return all_records, all_errors


def _to_records(models: EvalLangIDModels, all_prediction_counts, sample_count_threshold: int = 0, force_language_support: bool = True):
    records = []
    for model, data in all_prediction_counts.items():
        # print(f"Metrics for model: {model}")
        # pprint(data)
        metrics = _compute_all_languages_metrics(
            data['prediction_counts'],
            data['correct_prediction_counts'],
            data['incorrect_prediction_counts'],
            data['actual_sample_counts'],
            sample_count_threshold=sample_count_threshold,
        )
        for language, lang_metrics in metrics.items():
            # print(language)
            records.append({
                'model': model,
                'language iso639-3': language,
                'model supports language': True if force_language_support else models.model_supports_language(model, language),
                **lang_metrics,
                'samples/second (all languages)': data['samples/second']
            })

    return records



def _compute_metrics(total_preds, correct_preds, incorrect_preds, actual_sample_count):
    assert type(total_preds) == int
    assert type(correct_preds) == int
    assert type(actual_sample_count) == int
    assert type(incorrect_preds) == int
    precision = correct_preds / total_preds if total_preds > 0 else 0.0
    recall = correct_preds / actual_sample_count if actual_sample_count > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'predictions': total_preds, 'correct predictions': correct_preds, 'gt count': actual_sample_count,
            'precision': precision, 'recall': recall, 'f1': f1}

def _compute_all_languages_metrics(prediction_counts, correct_prediction_counts, incorrect_prediction_counts,
                                  actual_sample_counts, sample_count_threshold=0):
    metrics = {}
    for language in set(actual_sample_counts.keys()).union(set(prediction_counts.keys())):
        actual_sample_count = actual_sample_counts.get(language, 0)
        if actual_sample_count < sample_count_threshold:
            continue
        total_preds = prediction_counts.get(language, 0)
        correct_preds = correct_prediction_counts.get(language, 0)
        incorrect_preds = incorrect_prediction_counts.get(language, 0)
        metrics[language] = _compute_metrics(total_preds, correct_preds, incorrect_preds, actual_sample_count)

    return metrics


def _eval_all_models_on_dataset(
    models: EvalLangIDModels,
    dataset_label: str,
    dataset: Dataset,
    text_column: str,
    target_iso639_3_column: str,
    slow_models_to_skip: Optional[List[str]] = None,
    language_whitelist: Optional[List[str]] = None,
    batch_size: int=16,
    return_ytrue_and_ypred: bool = False,
) -> Tuple[Dict[str, Any], Counter]:
    error_counter = Counter()

    def register_error(error_string):
        if error_string not in error_counter:
            print(error_string)
        error_counter[error_string] += 1

    model_id_to_ytrue_and_ypred = {}

    all_prediction_counts = {}
    for langid_model_id in tqdm(models.all_model_ids, desc='Evaluating models on dataset ' + dataset_label):
        if slow_models_to_skip and langid_model_id in slow_models_to_skip:
            print("skipping slow", langid_model_id)
            continue
        prediction_counts = Counter()
        correct_prediction_counts = Counter()
        actual_sample_counts = Counter()
        incorrect_prediction_counts = Counter()
        start_time = time.perf_counter()

        # init
        model_id_to_ytrue_and_ypred[langid_model_id] = {
            'ytrue': [],
            'ypred': [],
        }

        with tqdm(total=len(dataset), desc=f'{langid_model_id}') as pbar:
            #seen_languages = set()
            for batch in dataset.iter(batch_size=batch_size):
                text_batch = batch[text_column]
                target_raw = batch[target_iso639_3_column]
                pbar.update(len(text_batch))

                target = []
                for i, tr in enumerate(target_raw):
                    #seen_languages.add(tr)
                    if tr is None or language_whitelist and tr not in language_whitelist:
                        target.append(None)
                    else:
                        try:
                            target.append(Lang(pt3=tr))
                        except DeprecatedLanguageValue as e:
                            register_error(f"Issue with dataset {dataset_label} specified language {target_raw}: {repr(e)}")
                            target.append(None)

                pred_batch, errors = models.identify_language_batch(text_batch, model=langid_model_id)
                for e in errors:
                    register_error(e)

                # batched check correctness, skipping None on target and using `und` for None on prediction
                for i, (pred, target) in enumerate(zip(pred_batch, target)):
                    target_iso693_3 = None if target is None else target.pt3
                    actual_sample_counts[target_iso693_3] += 1
                    if target_iso693_3 is None:
                        continue
                    if pred is None:
                        pred_iso693_3 = 'und'
                    elif len(pred.pt3) == 0:
                        register_error(f"{langid_model_id} output a language with no iso639-3 code: {pred}")
                        continue
                    else:
                        pred_iso693_3 = pred.pt3

                    prediction_counts[pred_iso693_3] += 1

                    # save individual predictions and targets
                    model_id_to_ytrue_and_ypred[langid_model_id]["ytrue"].append(target_iso693_3)
                    model_id_to_ytrue_and_ypred[langid_model_id]["ypred"].append(pred_iso693_3)

                    if target_iso693_3 == pred_iso693_3:
                        correct_prediction_counts[pred_iso693_3] += 1
                    else:
                        incorrect_prediction_counts[pred_iso693_3] += 1
                    # print(f"    Target: {target}, Pred: {pred}, Match: {target == pred}")
            #print("seen languages:", len(seen_languages), seen_languages)

        end_time = time.perf_counter()
        all_prediction_counts[langid_model_id] = {
            'actual_sample_counts': actual_sample_counts,
            'prediction_counts': prediction_counts,
            'correct_prediction_counts': correct_prediction_counts,
            'incorrect_prediction_counts': incorrect_prediction_counts,
            'samples/second': len(dataset) / (end_time - start_time)
        }

    if return_ytrue_and_ypred:
        return all_prediction_counts, error_counter, model_id_to_ytrue_and_ypred
    else:
        return all_prediction_counts, error_counter


def _check_iso693_3_column(dataset: Dataset, target_iso639_3_column: str, dataset_label: str):
    conform_counter = Counter()
    for record in tqdm(dataset, desc=f"Checking iso639-3 codes in dataset {dataset_label}"):
        langcode = record[target_iso639_3_column]
        if langcode is None:
            continue

        langcode_conformed, reason = conform_langcode_with_reason(langcode)
        if langcode_conformed != langcode:
            conform_counter[f"dataset has '{langcode}' -> conformed to {langcode_conformed}. Reason: \"{reason}\""] += 1

    if conform_counter:
        print(f" * Dataset {dataset_label} had non-conforming iso639-3 codes in column {target_iso639_3_column} that were conformed as follows:")
        for k, v in conform_counter.items():
            print(f"    - {v} occurrences: {k}")


def convert_and_conform_language(language_codes: Iterable[str]):
    language_codes = list(language_codes)
    errors = set()
    conformed_reasons = Counter()
    def to_iso639_3(lang_str):
        lang_code = lang_str.split('_')[0].split('-')[0]
        try:
            lang = Lang(lang_code)
            if len(lang.pt3) == 0:
                conformed_langcode = None
                reason = f"No ISO 639-3 code for language '{lang_code}'"
            else:
                conformed_langcode, reason = conform_langcode_with_reason(lang.pt3)
            if lang.pt3 != conformed_langcode:
                conformed_reasons[f"{lang_code} conformed to {conformed_langcode}. Reason: \"{reason}\""] += 1
            return conformed_langcode
        except (InvalidLanguageValue, DeprecatedLanguageValue) as e:
            errors.add(str(e))
            return None

    print("input: ", len(language_codes))
    result = [to_iso639_3(lc) for lc in language_codes]
    print("result: ", len(result))

    if errors:
        print("* Errors during language code conversion:")
        print("- " + "\n - ".join(errors))
    if conformed_reasons:
        print("* Language codes conformed:")
        for reason, count in conformed_reasons.items():
            print(f" - ({count} occurrences) {reason}")

    return result