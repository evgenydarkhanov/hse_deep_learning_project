# BM25 imports
from rank_bm25 import BM25Okapi
import numpy as np
import re
import pymorphy3

# embedding imports
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

# hybrid search imports
import pandas as pd


class BM25Search:
    def __init__(self, stopwords_path: str, documents: list[str]):

        with open(stopwords_path, 'r') as file:
            self._stop_words = {word.strip() for word in file}

        self._lemmatizer = pymorphy3.MorphAnalyzer()
        self._TOKEN_WORD = re.compile('[а-яА-ЯёЁ]{2,}')
        self._documents_tokenized = [self._one_sentence_preprocessing(doc, self._stop_words).split() for doc in documents]
        self._bm25_index = BM25Okapi(self._documents_tokenized)

    def _remove_lemmatize_preprocessing(self, sentence: str, words_to_remove: set) -> list:
        """ приводит к нижнему регистру, убирает числа и мусорные слова, лемматизирует """
        if isinstance(sentence, str):
            regex_tmp = re.findall(self._TOKEN_WORD, sentence.lower())
            remove_tmp = [self._lemmatizer.parse(token.strip())[0].normal_form for token in regex_tmp if token not in words_to_remove]
            return remove_tmp
        return []

    def _one_sentence_preprocessing(self, sentence: str, words_to_remove: set) -> str:
        """ применяет препроцессинги к одному предложению """
        sentence_words = self._remove_lemmatize_preprocessing(sentence, words_to_remove)
        result = ' '.join(sentence_words)
        return result

    def make_query(self, query: str, top: int = 5):
        """
        возвращает:
            - [{'corpus_id': id, 'score': score}, ...]
            - score - значения Okapi BM25 в диапазоне [0, 1]
        """

        query_cleaned = self._one_sentence_preprocessing(query, self._stop_words)
        query_tokenized = query_cleaned.split()

        scores = self._bm25_index.get_scores(query_tokenized)
        scores /= np.max(scores)

        top_k = np.argsort(scores)[::-1][:top]
        result = [{'corpus_id': i, 'score': scores[i]} for i in top_k]

        return result


class EmbeddingSearch:
    def __init__(self, embedder_path: str, documents: list[str], device):
        self._model = SentenceTransformer(
            model_name_or_path=embedder_path,
            device=device,
            local_files_only=True
        )
        self._documents_embedded = self._model.encode(documents)

    def make_query(self, query: str, top: int = 5):
        """
        возвращает:
            - [{'corpus_id': id, 'score': score}, ...]
            - score - нормированная на диапазон [0, 1] косинусная близость
        """
        query_emb = self._model.encode(query)
        result = util.semantic_search(query_emb, self._documents_embedded)[0][:top]
        for dct in result:
            dct['score'] = (dct['score'] + 1) / 2

        return result


class HybridSearch:
    def __init__(
        self,
        stopwords_path: str,
        embedder_path: str,
        documents: list[str],
        device,
        dataframe: pd.DataFrame
    ):
        self._bm_25 = BM25Search(stopwords_path, documents)
        self._embedding = EmbeddingSearch(embedder_path, documents, device)
        self._dataframe = dataframe

    def _scores_merging(
        self,
        *args,
        top: int,
        pooling: bool,
        threshold: float
    ) -> list[tuple[int | float]]:

        """
        объединяет результаты произвольного количества поисковиков
        оставляет уникальные ключи, суммирует скоры

        принимает: несколько словарей {id: score, ...}, {id: score, ...}; id - DataFrame index
        возвращает: список кортежей [(id, (score,)), ...]; id - DataFrame index

        if pooling is True:
            делает: Взвешенная сумма + Max pooling + топ-N
            возвращает: список кортежей [(id, (score, id)), ...]; id - DataFrame index

        if threshold is not None:
            возвращает то же самое, только score > threshold
        """

        tmp_result = {}

        # суммируем скоры с коэффициентами

        for arg in args:   # получаем словарь вида {index: (score, КОД УСЛУГИ), ...}
            for key, value in arg.items():
                service_code = self._dataframe.iloc[key, 1]
                if key not in tmp_result:
                    tmp_result[key] = (value, service_code)
                else:
                    new_value = value + tmp_result[key][0]
                    service_code = tmp_result[key][1]
                    tmp_result[key] = (new_value, service_code)

        # распаковка
        unpacked = [(key, value[0], value[1]) for key, value in tmp_result.items()]

        if pooling:        # получаем словарь вида {'КОД УСЛУГИ': (score, index), ...}
            result = {}
            for key, value in tmp_result.items():
                new_key = self._dataframe.iloc[key, 1]
                new_value = value[0]
                if new_key not in result:
                    result[new_key] = (new_value, key)
                else:
                    old_key, old_value = result[new_key]
                    if new_value > old_value:
                        result[new_key] = (new_value, key)

            # распаковка
            unpacked = [(value[1], value[0], key) for key, value in result.items()]

        # сортируем по итоговому значению скора и берём топ
        result = sorted(unpacked, key=lambda item: -item[1])[:top]

        if threshold is not None:
            result = list(filter(lambda elem: elem[1] > threshold, result))

        return result

    def make_query(
        self,
        query: str,
        top: int = 5,
        pooling: bool = False,
        threshold: float | None = None,
        weights={'bm_25': 0.5, 'embedding': 0.5},
    ):
        """
        возвращает спискок кортежей вида [(DataFrame_id: int, (score: float,)), ...]
                                или вида [(КОД УСЛУГИ: str, (score: float, DataFrame_id: int)), ...]
        """
        # проверяем веса
        assert abs(sum(weights.values()) - 1) < 0.01, "Сумма весов поисковиков должна быть равна 1"

        # получаем скоры поисковиков, умножаем на веса
        if weights['bm_25'] == 0:
            scores_bm_25 = {}

            embed_result = self._embedding.make_query(query, top)
            scores_embed = {dct['corpus_id']: dct['score'] * weights['embedding'] for dct in embed_result}

        if weights['embedding'] == 0:
            scores_embed = {}

            bm_25_result = self._bm_25.make_query(query, top)
            scores_bm_25 = {dct['corpus_id']: dct['score'] * weights['bm_25'] for dct in bm_25_result}

        else:
            bm_25_result = self._bm_25.make_query(query, top)
            embed_result = self._embedding.make_query(query, top)

            scores_bm_25 = {dct['corpus_id']: dct['score'] * weights['bm_25'] for dct in bm_25_result}
            scores_embed = {dct['corpus_id']: dct['score'] * weights['embedding'] for dct in embed_result}

        # объединяем результаты в требуемом формате
        result = self._scores_merging(
            scores_bm_25,
            scores_embed,
            top=top,
            pooling=pooling,
            threshold=threshold
        )

        return result