import numpy as np


class RetrieverMetrics:
    def __init__(self, y_true, y_pred, top):
        self.y_true = y_true
        self.y_pred = y_pred
        self.top = top

    def p_at_k(self, y_true, y_pred) -> float:
        """
        Computes precision at k

        Parameters
        ----------
        y_true: Iterable
                A collection of all relevant labels
        y_pred: Iterable
                A collection of predicted labels
        """
        if not y_pred:
            return 0.0
        count = 0
        actual = set(y_true)
        for elem in y_pred:
            if elem in actual:
                count += 1
        score = count / len(y_pred)
        return score

    def compute_theo_max(self, top: int) -> float:
        """
        Computes theoretical maximum for every task
        """
        theo_array = [0 for i in range(top)]
        theo_array[0] = 1
        theo_true = [1]
        score = 0.0
        for i in range(1, top + 1):
            score += self.p_at_k(theo_true, theo_array[:i])

        score /= top
        return score

    def ap_at_k(self, y_true, y_pred, top: int) -> float:
        """
        Computes average precision at k

        Parameters
        ----------
        y_true: Iterable
                A collection of all relevant labels
        y_pred: Iterable
                A collection of predicted labels
        top:    int
                The maximum number of predicted elements
        """
        score = 0.0
        actual_length = min(top, len(y_pred))

        for i in range(1, actual_length + 1):
            score += self.p_at_k(y_true, y_pred[:i])

        theo_max = self.compute_theo_max(actual_length)
        norm_score = score / theo_max

        norm_score /= actual_length
        score /= actual_length
        # print(f"{actual_length=}, {score=}, {theo_max=}, {norm_score=}")
        return score, norm_score

    def map_at_k(self, y_true, y_pred, top: int) -> tuple[float]:
        """
        Computes mean average precision at k

        Parameters
        ----------
        y_true: Iterable[Iterable]
                A collection of collections of all relevant labels
        y_pred: Iterable[Iterable]
                A collection of collections of predicted labels
        top:    int
                The maximum number of predicted elements
        """
        # theo_max = self.compute_theo_max(top)
        # score = np.mean([self.ap_at_k(actual, pred, top) for actual, pred in zip(y_true, y_pred)])
        # norm_score = score / theo_max

        aps_at_k = [self.ap_at_k(actual, pred, top) for actual, pred in zip(y_true, y_pred)]

        score = np.mean([elem[0] for elem in aps_at_k])
        norm_score = np.mean([elem[1] for elem in aps_at_k])

        return score, norm_score

    def rr_at_k(self, y_true, y_pred) -> float:
        """
        Computes reciprocal rank at k

        Parameters
        ----------
        y_true: Iterable[Iterable]
                A collection of all relevant labels
        y_pred: Iterable[Iterable]
                A collection of predicted labels
        """
        if not y_pred:
            return 0.0
        actual = set(y_true)
        score = 0.0
        for idx, elem in enumerate(y_pred):
            if elem in actual:
                score = 1 / (idx + 1)
                return score
        return score

    def mrr_at_k(self, y_true, y_pred) -> float:
        """
        Computes reciprocal rank at k

        Parameters
        ----------
        y_true: Iterable[Iterable]
                A collection of collections of all relevant labels
        y_pred: Iterable[Iterable]
                A collection of collections of predicted labels
        """
        score = np.mean(
            [self.rr_at_k(actual, pred) for actual, pred in zip(self.y_true, self.y_pred)]
        )
        return score

    def report(self):
        map_k, n_map_k = self.map_at_k(self.y_true, self.y_pred, self.top)
        mrr_k = self.mrr_at_k(self.y_true, self.y_pred)
        # print(f"MAP@K = {map_k:.4f},\t norm_MAP@K = {n_map_k:.4f}\tMRR@K = {mrr_k:.4f}")
        return map_k, n_map_k, mrr_k