from rank_bm25 import BM25Okapi
import numpy as np
import re
import pymorphy3


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