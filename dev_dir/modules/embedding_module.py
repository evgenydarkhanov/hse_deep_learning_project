import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np


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