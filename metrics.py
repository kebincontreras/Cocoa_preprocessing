import numpy as np

from sklearn.metrics import confusion_matrix


def compute_metric_params(matrix):
    shape = np.shape(matrix)
    number = 0
    add = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        add += np.sum(matrix[i, :]) * np.sum(matrix[:, i])

    return AA, add, number


def overall_accuracy(matrix, number):
    return number / np.sum(matrix)


def average_accuracy(AA):
    return np.mean(AA)


def kappa(OA, matrix, add):
    pe = add / (np.sum(matrix) ** 2)
    return (OA - pe) / (1 - pe)


def print_results(classifier_name, dataset_name, dict_metrics):
    print('#================================================#')
    print(f'Classifier: {classifier_name}, Dataset: {dataset_name}')
    for name, metrics in dict_metrics.items():
        out_print = f'{name:<5} -> '
        for n_metric, metric in metrics.items():
            out_print += f'{n_metric}: {100 * metric:.2f} '
        print(out_print)
    print('#================================================#')
