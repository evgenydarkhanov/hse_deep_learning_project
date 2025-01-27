## Импортируемые модули для проекта "Чат-бот для МФЦ"

- `bm_25_module.py`, `bm_25_requirements.txt`

Реализует поиск по векторам Okapi BM25. Требуются стоп-слова для языка поиска.

Пример использования:

```python
pip install -r ./bm_25_requirements.txt
from bm_25_module import *

stopwords_path = 'PATH_TO_STOPWORDS'
documents_to_search = ['YOUR_DOC_1', 'YOUR_DOC_2', 'YOUR_DOC_3']    # type must be list[str]

bm25_searcher = BM25Search(stopwords_path, documents_to_search)
scores = bm25_searcher.make_query('YOUR_QUERY')
```

- `embedding_module.py`, `embedding_requirements.txt`

Реализует поиск по эмбеддингам и косинусной близости. Эмбеддер может быть произвольным, однако должен быть совместимым с библиотекой `SentenceTransformers`.

Пример использования:

```python
pip install -r ./embedding_requirements.txt
from embedding_module import *

embedder_path = "YOUR_EMBEDDER_PATH"    # your embedder must be compatible with the SentenceTransformers library
documents_to_search = ['YOUR_DOC_1', 'YOUR_DOC_2', 'YOUR_DOC_3']    # type must be list[str]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb_searcher = EmbeddingSearch(embedder_path, documents_to_search, device)
scores = emb_searcher.make_query('YOUR_QUERY')
```

- `hybrid_search_module.py`, `hybrid_search_requirements.txt`

Реализует гибридный поиск по эмбеддингам и векторам Okapi BM25.

- Требуются стоп-слова для языка поиска
- Эмбеддер может быть произвольным, однако должен быть совместимым с библиотекой `SentenceTransformers`
- Требуется объект `pandas.DataFrame`, из которого взяты данные для поиска
- `pandas.DataFrame` необходим для реализации механизма "Взвешенная сумма + Max pooling + топ-N"
- через параметр `weights={'bm_25': 0.5, 'embedding': 0.5}` в `HybridSearcher.make_query()` можно варьировать важность каждого из поисковиков. Веса в сумме должны быть равны единице. Если какой-то из весов равен нулю, то соответствующий поиск не производится

Пример использования:

```python
pip install -r ./hybrid_search_requirements.txt
from hybrid_search_module import *

stopwords_path = 'PATH_TO_STOPWORDS'
documents_to_search = ['YOUR_DOC_1', 'YOUR_DOC_2', 'YOUR_DOC_3']    # type must be list[str]

embedder_path = "YOUR_EMBEDDER_PATH"    # your embedder must be compatible with the SentenceTransformers library

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hybrid_searcher = HybridSearch(stopwords_path, embedder_path, documents_to_search, device, dataframe)
scores = hybrid_searcher.make_query('YOUR_QUERY')
```

- `modules_test.ipynb`

Ноутбук с примерами использования модулей.

### Оценивание работы поисковиков

- `metrics_module.py`

Содержит в себе метрики `MAP@K` и `MRR@K`, используемые для оценивания качества работы поисковиков. Возвращает три значения:

- `MAP@K` - [0, 1], больше - лучше. Никогда не достигнет единицы из-за особенностей решаемой задачи. **Показывает насколько релевантен список рекомендуемых элементов**.
- `norm_MAP@K` - [0, 1], больше - лучше. То же самое, только масштабировано к нашей задаче, поэтому может достигнуть единицы.
- `MRR@K` - [0, 1], больше - лучше. Может достигнуть единицы. **Показывает вероятность нахождения необходимого элемента первым в списке**.

Пример использования:

```python
from metrics_module import *

TOP = 4

y_pred = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    ['not', 'not', 'relevant', 'not']
]
y_true = [
    [1],
    [1],
    [1],
    ['relevant']
]

metrics = RetrieverMetrics(y_true, y_pred, TOP)
metrics.report()
```

- `retriever_metrics.ipynb` - ноутбук с результатами подбора гиперпараметров для поисковиков

Поскольку все эмбеддеры дают схожее качество, то предлагается использовать USER-bge-m3 с весом 0.5.

- обучен на русском языке
- работает быстрее товарищей
- хорошие метрики у энкодечки
- выпущен, вроде, летом 2024

[rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2)

| BM25 | rubert-tiny2 | MAP@5 | norm_MAP@5 | MRR@5 |
| :--: | :--: | :--: | :--: | :--: |
| 1.0 | 0.0 | 0.4280 | 0.9280 | 0.9254 |
| 0.9 | 0.1 | 0.4262 | 0.9334 | 0.9314 |
| 0.8 | 0.2 | 0.4262 | 0.9334 | 0.9314 |
| 0.7 | 0.3 | 0.4262 | 0.9334 | 0.9314 |
| 0.6 | 0.4 | 0.4262 | 0.9334 | 0.9314 |
| **0.5** | **0.5** | **0.4262** | **0.9334** | **0.9314** |
| 0.4 | 0.6 | 0.4262 | 0.9334 | 0.9314 |
| 0.3 | 0.7 | 0.4262 | 0.9334 | 0.9314 |
| 0.2 | 0.8 | 0.4262 | 0.9334 | 0.9314 |
| 0.1 | 0.9 | 0.4262 | 0.9334 | 0.9314 |
| 0.0 | 1.0 | 0.4262 | 0.9334 | 0.9314 |

[ruBert-base](https://huggingface.co/ai-forever/ruBert-base)

| BM25 | ruBert-base | MAP@5 | norm_MAP@5 | MRR@5 |
| :--: | :--: | :--: | :--: | :--: |
| 1.0 | 0.0 | 0.4280 | 0.9280 | 0.9254 |
| 0.9 | 0.1 | 0.4257 | 0.9313 | 0.9290 |
| 0.8 | 0.2 | 0.4257 | 0.9313 | 0.9290 |
| 0.7 | 0.3 | 0.4257 | 0.9313 | 0.9290 |
| 0.6 | 0.4 | 0.4257 | 0.9313 | 0.9290 |
| **0.5** | **0.5** | **0.4257** | **0.9313** | **0.9290** |
| 0.4 | 0.6 | 0.4257 | 0.9313 | 0.9290 |
| 0.3 | 0.7 | 0.4257 | 0.9313 | 0.9290 |
| 0.2 | 0.8 | 0.4257 | 0.9313 | 0.9290 |
| 0.1 | 0.9 | 0.4257 | 0.9313 | 0.9290 |
| 0.0 | 1.0 | 0.4257 | 0.9313 | 0.9290 |

[bi-encoder](https://huggingface.co/DiTy/bi-encoder-russian-msmarco)

| BM25 | bi-encoder | MAP@5 | norm_MAP@5 | MRR@5 |
| :--: | :--: | :--: | :--: | :--: |
| 1.0 | 0.0 | 0.4280 | 0.9280 | 0.9254 |
| 0.9 | 0.1 | 0.4268 | 0.9334 | 0.9314 |
| 0.8 | 0.2 | 0.4268 | 0.9334 | 0.9314 |
| 0.7 | 0.3 | 0.4268 | 0.9334 | 0.9314 |
| 0.6 | 0.4 | 0.4268 | 0.9334 | 0.9314 |
| **0.5** | **0.5** | **0.4268** | **0.9334** | **0.9314** |
| 0.4 | 0.6 | 0.4268 | 0.9334 | 0.9314 |
| 0.3 | 0.7 | 0.4268 | 0.9334 | 0.9314 |
| 0.2 | 0.8 | 0.4268 | 0.9334 | 0.9314 |
| 0.1 | 0.9 | 0.4268 | 0.9334 | 0.9314 |
| 0.0 | 1.0 | 0.4268 | 0.9334 | 0.9314 |

[USER-bge-m3](https://huggingface.co/deepvk/USER-bge-m3)

| BM25 | USER-bge-m3 | MAP@5 | norm_MAP@5 | MRR@5 |
| :--: | :--: | :--: | :--: | :--: |
| 1.0 | 0.0 | 0.4280 | 0.9280 | 0.9254 |
| 0.9 | 0.1 | 0.4265 | 0.9334 | 0.9314 |
| 0.8 | 0.2 | 0.4265 | 0.9334 | 0.9314 |
| 0.7 | 0.3 | 0.4265 | 0.9334 | 0.9314 |
| 0.6 | 0.4 | 0.4265 | 0.9334 | 0.9314 |
| **0.5** | **0.5** | **0.4265** | **0.9334** | **0.9314** |
| 0.4 | 0.6 | 0.4265 | 0.9334 | 0.9314 |
| 0.3 | 0.7 | 0.4265 | 0.9334 | 0.9314 |
| 0.2 | 0.8 | 0.4265 | 0.9334 | 0.9314 |
| 0.1 | 0.9 | 0.4265 | 0.9334 | 0.9314 |
| 0.0 | 1.0 | 0.4265 | 0.9334 | 0.9314 |
