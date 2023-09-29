import numpy as np
from collections import defaultdict
import torch


class Evaluator:

    """
    A class for evaluating model performance.
    it has been used only to measure the performance of the METER ML dataset 
    Attributes:
    - classes_labels: List[str]
        The labels of the classes.
    - false_positives: List[int]
        The number of false positives for each class.
    - success: List[int]
        The number of successful predictions for each class.
    - totals: List[int]
        The total number of samples for each class.
    - ignore_index: int
        The index of the class to ignore during evaluation.

    Methods:
    - fit(preds, labels): Update the evaluation metrics based on the predicted and true labels.
    - finish(): Print the evaluation metrics and return the total accuracy.
    """

    classes_labels = [
        "CAFOs",
        "Landfills",
        "Mines",
        "Negative",
        "ProcessingPlants",
        "RefineriesAndTerminals",
        "WWTreatment",
    ]
    def __init__(self, ignore_index=3):
        self.false_positives = [0 for _ in self.classes_labels]
        self.success = [0 for _ in self.classes_labels]
        self.totals = [0 for _ in self.classes_labels]
        self.ignore_index = ignore_index

    def fit(self, preds, labels):
        for i, _ in enumerate(Evaluator.classes_labels):
            if i == self.ignore_index:
                continue
            mask = labels == i
            self.false_positives[i] += torch.sum(preds[~mask] == i).item()
            self.success[i] += torch.sum(preds[mask] == i).item()
            self.totals[i] += mask.sum().item()

    def finish(self):
        max_title_width = max(len(cls) for cls in Evaluator.classes_labels)
        print(" " * (max_title_width+1), "ACC   ", "PREC  ", "RECALL")
        
        def div_and_fmt(a, b):
            if b == 0:
                return "NaN"
            res = a / b
            return f"{res:03.02f}"

        for i, cls in enumerate(Evaluator.classes_labels):
            if i == self.ignore_index:
                continue
            pr = div_and_fmt(100. * self.success[i], (self.success[i] + self.false_positives[i]))
            recall = div_and_fmt(100. * self.success[i], self.totals[i])
            #acc = div_and_fmt(100. * self.success[i], (self.success[i] + self.false_positives[i] + self.totals[i] - self.success[i]))
            msg = " " * 6 + f" {pr: >5}% {recall: >5}% "
            print(" " * (max_title_width - len(cls)), cls, msg)

        totals = sum(self.totals)
        if totals == 0:
            raise Exception("totals is 0")
        else:
            total_acc = 100. * sum(self.success) / sum(self.totals)

        print(" " * (max_title_width - 5), "TOTAL", f"{total_acc:3.02f}%")
        print("==")
        return total_acc




class ClasswiseAccuracy(object):
    """
    A class for calculating class-wise accuracy.
    this has been used for the METER ML and the DFC2020 dataset.
    Note that if you use this class to measure the performance of the METER ML dataset 
    you'll have to take into account the class index 3 which represend the background 
    so if you set num_classes to 7 (for the METER ML) you can get the real accuracy by 
    adding each accuracy except the third one divided by 6
    Attributes:
    - num_classes: int
        The number of classes.
    - tp_per_class: defaultdict
        The number of true positives for each class.
    - count_per_class: defaultdict
        The count of samples for each class.
    - count: int
        The total count of samples.
    - tp: int
        The total count of true positives.

    Methods:
    - add_batch(y, y_hat): Update the class-wise accuracy based on the true labels (y) and predicted labels (y_hat).
    - get_classwise_accuracy(): Calculate the class-wise accuracy.
    - get_average_accuracy(): Calculate the average accuracy across all classes.
    - get_overall_accuracy(): Calculate the overall accuracy.
    
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp_per_class = defaultdict(int)
        self.count_per_class = defaultdict(int)
        self.count = 0
        self.tp = 0

    def add_batch(self, y, y_hat):
        for true, pred in zip(y, y_hat):
            self.count_per_class["class_" + str(true.item())] += 1
            self.count += 1
            if true == pred:
                self.tp_per_class["class_" + str(true.item())] += 1
                self.tp += 1

    def get_classwise_accuracy(self):
        return {k: self.tp_per_class[k] / count for k, count in self.count_per_class.items()}

    def get_average_accuracy(self):
        cw_acc = self.get_classwise_accuracy()
        return np.mean(list(cw_acc.values()))

    def get_overall_accuracy(self):
        return self.tp / self.count

