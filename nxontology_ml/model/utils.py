import warnings

import numpy as np
import scipy

label_to_int = {
    "01-disease-subtype": 0,
    "02-disease-root": 1,
    "03-disease-area": 2,
}
one_h_enc = {
    "01-disease-subtype": np.array([1, 0, 0]),
    "02-disease-root": np.array([0, 1, 0]),
    "03-disease-area": np.array([0, 0, 1]),
}
CLASS_COUNTS = np.array([7524, 5236, 1603])  # Counts from whole training set
CLASS_WEIGHTS = CLASS_COUNTS / np.sum(CLASS_COUNTS)
BIASED_CLASS_WEIGHTS = np.array([0.25, 0.25, 0.5])


def biased_sample_weights(y: list[str]) -> np.ndarray:
    return np.array([BIASED_CLASS_WEIGHTS[label_to_int[lbl]] for lbl in y])


def pad(a: np.array, class_weight: np.array, fill_with: float = 0) -> np.array:
    # If a class is missing form the batch, the weights can be the wrong dim
    return np.pad(
        a,
        pad_width=((0, 0), (0, len(class_weight) - a.shape[1])),
        constant_values=fill_with,
    )


def mean_absolute_error(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    class_weight: np.ndarray | None = None,
) -> float:
    class_weight = class_weight if class_weight is not None else BIASED_CLASS_WEIGHTS
    y_probas_padded = pad(y_probas, class_weight)
    m = np.mean(np.dot(np.abs(y_true - y_probas_padded), class_weight))
    assert isinstance(m, float)
    return m


class BiasedMaeMetric:
    """
    MAE Metric using a custom biased weight for the classes

    Examples: https://catboost.ai/en/docs/concepts/python-usages-examples#custom-loss-function-eval-metric

    FIXME:
    ```
    catboost/core.py:2268: UserWarning: Can't optimize method "evaluate" because self argument is used
        _check_train_params(params)
    ```
    """

    def is_max_optimal(self) -> bool:
        # Returns whether great values of metric are better
        return False

    @classmethod
    def evaluate(
        cls,
        approxes: tuple[np.ndarray, ...],
        target: np.ndarray,
        weight: np.ndarray | None,
    ) -> tuple[float, int]:
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.
        # weight parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # Returns pair (error, weights sum)
        if weight is not None:
            warnings.warn(
                "`BiasedMaeMetric` ignores sample weights and uses BIASES_CLASS_WEIGHTS.",
                stacklevel=1,
            )

        size = target.size

        # Normalize predictions
        approxes_t = np.transpose(np.stack(approxes))
        norm_pred = scipy.special.softmax(approxes_t, axis=1)

        # Labels to one hot encode
        clipped_target = np.clip(
            target, a_min=0.0, a_max=3.0, casting="unsafe", dtype=np.int8
        )
        one_hot_target = np.zeros((size, len(BIASED_CLASS_WEIGHTS)), dtype=np.int8)
        one_hot_target[np.arange(size), clipped_target] = 1

        # MAE
        abs_diff_padded = pad(np.abs(one_hot_target - norm_pred), BIASED_CLASS_WEIGHTS)
        sum_mae = np.sum(np.dot(abs_diff_padded, BIASED_CLASS_WEIGHTS))
        weight_mae = size
        return sum_mae, weight_mae

    def get_final_error(self, error: float, weight: int) -> float:
        # Returns final value of metric based on error and weight
        return error / (weight + 1e-38)
