import hashlib
import json
import os
from collections import defaultdict
from typing import List, Union, Tuple, Optional

import regex
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue, DeprecatedLanguageValue
import fasttext
from huggingface_hub import hf_hub_download


# https://huggingface.co/datasets/laurievb/OpenLID-v2/blob/main/scripts/tools/openlid_normer.py
NONWORD_REPLACE_STR = r"[^\p{Word}\p{Zs}]|\d"
NONWORD_REPLACE_PAT = regex.compile(NONWORD_REPLACE_STR)
SPACE_PAT = regex.compile(r"\s\s+")
def openlid_normer_clean_line(line):
    """simple language-agnostic cleaning"""
    text = line.strip().replace("\n", " ").lower()  # remove whitespace, apply lowercase
    text = regex.sub(NONWORD_REPLACE_PAT, "", text)  # either (not a word nor a space) or (is digit)
    text = regex.sub(SPACE_PAT, " ", text)  # squeeze whitespace
    return text



class EvalLangIDModels:
    def __init__(
            self,
            do_conform_langcode: bool = True,
            models_to_register: Optional[List[str]] = None,
            supported_languages_matrix_csv_path: Optional[str] = None,
        ):
        """ Initialize and register language identification models.

        supported_languages_matrix_csv_path: path to CSV file with column 'language iso693-3' and then one column per
            model id. rows are language (iso639-3), with 1/0 in columns for models indicating support for that language.
            eg :
            language iso639-3,cld2,gcld3,GlotLID,OpenLID-v2,pyfranc,AfroLID,fasttext,funlangid
            eng,1,1,1,1,1,1,1,1
            ton,1,0,0,1,0,1,1,0
            ...
        """
        if models_to_register is None:
            models_to_register = ['cld2', 'gcld3', 'GlotLID', 'OpenLID-v2', 'pyfranc', 'AfroLID', 'fasttext', 'funlangid']
        self.eval_models_dict = {}
        self.enable_conform_langcode = do_conform_langcode

        if 'cld2' in models_to_register:
            self._register_cld2()
        if 'gcld3' in models_to_register:
            self._register_gcld3()
        if 'GlotLID' in models_to_register:
            self._register_glotlid()
        if 'OpenLID-v2' in models_to_register:
            self._register_openlidv2()
        if 'pyfranc' in models_to_register:
            self._register_pyfranc()
        if 'AfroLID' in models_to_register:
            self._register_afrolid()
        if 'fasttext' in models_to_register:
            self._register_fasttext()
        if 'funlangid' in models_to_register:
            self._register_funlangid()

        if supported_languages_matrix_csv_path is None:
            self.supported_languages = {}
        else:
            self.supported_languages = defaultdict(set)
            import pandas as pd
            df = pd.read_csv(supported_languages_matrix_csv_path)
            for _, row in df.iterrows():
                for model_name in self.eval_models_dict.keys():
                    if row[model_name] == 1:
                        self.supported_languages[model_name].add(row['language iso639-3'])

    @property
    def all_model_ids(self) -> List[str]:
        return list(self.eval_models_dict.keys())


    def identify_language_batch(self, text: Union[str, List[str]], model="cld2") -> Tuple[List[Optional[Lang]], List[str]]:
        if type(text) is str:
            text = [text]
        elif type(text) is not list:
            raise TypeError("text must be a str or a list of str")
        text_cleaned = [openlid_normer_clean_line(t) for t in text]
        eval_func = self.eval_models_dict[model]
        iso693_3_codes = eval_func(text_cleaned)

        langs = []
        errors = []
        for code in iso693_3_codes:
            try:
                langs.append(_iso693_to_lang_maybe(code))
            except DeprecatedLanguageValue as e:
                errors.append(f"{model} output a deprecated language code: '{repr(e)}'")
                langs.append(None)
                continue
            except InvalidLanguageValue as e:
                errors.append(f"{model} output an invalid language code: '{repr(e)}'")
                langs.append(None)

        return langs, errors


    def model_supports_language(self, model_id: str, language_iso639_3: str) -> bool:
        """ return whether the given model supports the given language """
        return language_iso639_3 in self.supported_languages[model_id]


    def register_model(self, model_id: str, eval_func):
        self.eval_models_dict[model_id] = eval_func
        print("registered model", model_id)


    def _conform_langcode(self, langcode: str) -> Union[str,None]:
        """ apply common language code conformation to make compatible with iso639-3 """
        if not self.enable_conform_langcode:
            return langcode

        return conform_langcode(langcode)


    def _register_cld2(self):
        import pycld2 as cld2

        def _eval_cld2(text) -> Union[str,None]:
            isReliable, textBytesFound, details = cld2.detect(text, isPlainText=True)
            lang_code_with_script = details[0][1]
            lang_code_no_script = lang_code_with_script.split('-')[0]
            if lang_code_no_script == 'un':
                return None
            elif lang_code_no_script == 'xx':
                return None
            elif lang_code_no_script == 'zzp':
                return None
            else:
                return self._conform_langcode(lang_code_no_script)

        def _eval_cld2_batch(texts: List[str]) -> List[Union[str,None]]:
            return _eval_batch_generic(texts, _eval_cld2)

        self.register_model('cld2', _eval_cld2_batch)

    def _register_gcld3(self):
        try:
            import gcld3
        except ImportError:
            print("gcld3 not installed, skipping")
            return
        detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                                max_num_bytes=4096)

        def _eval_gcld3(text: str) -> Union[str,None]:
            result = detector.FindLanguage(text)
            lang_code = result.language.split('-')[0]
            if lang_code == 'und':
                return None
            else:
                return self._conform_langcode(lang_code)

        def _eval_gcld3_batch(texts: List[str]) -> List[Union[str, None]]:
            return _eval_batch_generic(texts, _eval_gcld3)
            
        self.register_model('gcld3', _eval_gcld3_batch)
            

    def _register_glotlid(self):
        # model.bin is the latest version always
        glotlid_model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        glotlid_model = fasttext.load_model(glotlid_model_path)

        def _eval_glotlid_batch(text) -> List[str]:
            labels = _ft_predict_labels(glotlid_model, text)
            return self._process_generic_fasttext_output(model_id='GlotLID', label_lists=labels)
        self.register_model('GlotLID', _eval_glotlid_batch)


    def _register_openlidv2(self):
        openlidv2_model_path = hf_hub_download(repo_id="laurievb/OpenLID-v2", filename="model.bin")
        openlidv2_model = fasttext.load_model(openlidv2_model_path)

        def _eval_openlidv2_batch(text: List[str]) -> List[str]:
            labels = _ft_predict_labels(openlidv2_model, text)
            return self._process_generic_fasttext_output(model_id='OpenLID-v2', label_lists=labels)

        self.register_model('OpenLID-v2', _eval_openlidv2_batch)


    def _register_pyfranc(self):
        from pyfranc import franc

        def _eval_pyfranc_batch(text: List[str]) -> List[str]:
            langcode = [franc.lang_detect(t) for t in text]
            return [r[0][0] for r in langcode]

        self.register_model('pyfranc', _eval_pyfranc_batch)


    def _register_afrolid(self):
        from transformers import pipeline
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        afrolid = pipeline("text-classification", model='UBC-NLP/afrolid_1.5', device=device)

        def _eval_afrolid_batch(text: List[str]) -> List[Optional[str]]:
            tokenizer_kwargs = {'truncation': True, 'max_length': 512}
            result = afrolid(text, **tokenizer_kwargs)
            language = []
            for r in result:
                label = r['label']
                if label == 'nan_lang':
                    language.append(None)
                else:
                    language.append(self._conform_langcode(label))
            return language

        self.register_model('AfroLID', _eval_afrolid_batch)

    def _register_fasttext(self):
        fasttext_model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        fasttext_model = fasttext.load_model(fasttext_model_path)

        def _eval_fasttext_batch(text: List[str]) -> List[str]:
            labels = _ft_predict_labels(fasttext_model, text)
            return self._process_generic_fasttext_output(model_id='fasttext', label_lists=labels)

        self.register_model('fasttext', _eval_fasttext_batch)

    def _register_funlangid(self):
        from fun_langid import FunLangID
        fun_langid = FunLangID()

        def _eval_funlangid_batch(text: List[str]) -> List[str]:
            langcodes = []
            for t in text:
                funlangid_output = fun_langid.predict_top(t)
                # output is BCP-47 'lang-script'
                langcode = funlangid_output.split('-')[0]
                if langcode == 'und':
                    langcodes.append(None)
                else:
                    langcodes.append(self._conform_langcode(langcode))
            return langcodes

        self.register_model('funlangid', _eval_funlangid_batch)

    def _register_dspy_module(self, model_name: str, batch_size: int = 100, n_threads: int = 1, cache_dir: str = "./cache"):
        from llm_eval.dspy_langid_module import DSPyLangIDModule
        from llm_eval.dspy_utils import batched_predict_with_dspy

        langid_module = DSPyLangIDModule()

        os.makedirs(cache_dir, exist_ok=True)

        def _eval_dspy_batch(text: List[str]) -> List[str]:
            hash_str = hashlib.sha256(json.dumps(text, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
            
            df = batched_predict_with_dspy(
                input_examples=[dspy.Example(text=t).with_inputs("text") for t in text],
                module=langid_module,
                batch_size=batch_size,
                cache_base_file_path=os.path.join(cache_dir, "cache_" + model_name.replace("/", "_") + "_" + hash_str),
                skip_llm_confirmation=True,
                n_threads=n_threads,
            )
            return df["language_iso639_3"].values.tolist()
                    
        self.register_model(f'dspy_{model_name}', _eval_dspy_batch)


    def _process_generic_fasttext_output(self, model_id: str, label_lists: list) -> List[Union[str,None]]:
        langcodes = []
        for results in label_lists:
            langtext = results[0]
            if not langtext.startswith("__label__"):
                raise ValueError(f"Unexpected label format from {model_id} model: " + langtext)
            langcode = langtext.split("__")[2].split('_')[0]
            langcodes.append(self._conform_langcode(langcode))
        return langcodes


def _ft_predict_labels(ft_model, texts: List[str]) -> list:
    """Return ``[[label, ...], ...]`` regardless of fasttext-wheel vs fasttext-predict.

    ``fasttext-wheel`` returns ``(labels, probs)`` from ``predict``;
    ``fasttext-predict`` returns a bare ``labels`` list but ships a broken
    ``predict`` wrapper on Python 3.13. We sidestep it by calling the
    underlying C++ binding ``multilinePredict`` when available.
    """
    mp = getattr(ft_model.f, "multilinePredict", None)
    if mp is not None:
        prepared = [t if t.endswith("\n") else t + "\n" for t in texts]
        result = mp(prepared, 1, 0.0, "strict")
        if isinstance(result, tuple) and len(result) == 2:
            return list(result[0])
        return list(result)
    predicted = ft_model.predict(list(texts))
    if isinstance(predicted, tuple) and len(predicted) == 2:
        return list(predicted[0])
    return list(predicted)


def _eval_batch_generic(texts, callback) -> List[Union[str,None]]:
    results = []
    for text in texts:
        res = callback(text)
        results.append(res)
    return results




# identifier

def _iso693_to_lang_maybe(iso693_3_code: Union[str,None]) -> Union[Lang,None]:
    if iso693_3_code is None:
        return None
    lang = Lang(iso693_3_code)
    return lang

def conform_langcode(langcode: str) -> Union[str,None]:
    conformed_langcode, _ = conform_langcode_with_reason(langcode)
    return conformed_langcode

def conform_langcode_with_reason(langcode: str) -> Tuple[Union[str,None], Union[str,None]]:
    # Deprecation messages are copied verbatim from iso639-lang python package deprecation errors
    if langcode == 'jw':
        return 'jav', 'As of 2001-08-13, [jw] for Javanese is deprecated due to deprecated. Use [jv] instead.'
    elif langcode == 'bh':
        return 'bih', 'As of 2021-05-25, [bh] for Bihari languages is deprecated due to deprecated. Two-letter identifier bh deprecated in ISO 639-1; use of three-letter identifier bih for Bihari languages is favored.'
    elif langcode == 'iw':
        return 'heb', 'As of 1989-03-11, [iw] for Hebrew is deprecated due to deprecated. Use [he] instead.'
    elif langcode == 'ajp':
        return 'apc', 'As of 2023-01-20, [ajp] for South Levantine Arabic is deprecated due to merge. Use [apc] instead.'
    elif langcode == 'eml':
        return None, 'As of 2009-01-16, [eml] for Emiliano-Romagnolo is deprecated due to split. Split into Emilian [egl] and Romagnol [rgn].'
    elif langcode == 'tpw':
        return 'tpn', 'As of 2023-01-20, [tpw] for Tupí is deprecated due to duplicate. Use [tpn] instead.'
    elif langcode == 'oto':
        return None, "No iso639-3 code: Lang(name='Otomian languages', pt1='', pt2b='oto', pt2t='oto', pt3='', pt5='oto')"
    elif langcode == 'ber':
        return 'tzm', "No iso639-3 code: Lang(name='Berber languages', pt1='', pt2b='ber', pt2t='ber', pt3='', pt5='ber') -> use Central Atlas Tamazight [tzm])"
    elif langcode == 'ngo':
        return None, 'As of 2021-01-15, [ngo] for Ngoni is deprecated due to split. Split into Ngoni (Tanzania) [xnj] and Ngoni (Mozambique) [xnq].'
    elif langcode == 'kzj':
        return 'dtp', 'As of 2016-01-15, [kzj] for Coastal Kadazan is deprecated due to merge. Use [dtp] instead.'
    elif langcode == 'dan':
        return None, 'As of 2013-01-23, [daf] for Dan is deprecated due to split. Split into Dan [dnj] and Kla-Dan [lda].'
    elif langcode == 'kxu':
        return None, 'As of 2020-01-23, [kxu] for Kui (India) is deprecated due to split. Split into [dwk] Dawik Kui and [uki] Kui (India).'
    elif langcode == 'nah':
        return None, "No iso639-3 code: Lang(name='Nahuatl languages', pt1='', pt2b='nah', pt2t='nah', pt3='', pt5='nah')"
    elif langcode == 'bih':
        return None, "No iso639-3 code: Lang(name='Bihari languages', pt1='', pt2b='bih', pt2t='bih', pt3='', pt5='bih')"

    # seems ok
    return langcode, None