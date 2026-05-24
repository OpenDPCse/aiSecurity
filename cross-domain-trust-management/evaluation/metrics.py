from typing import Dict, List

from core.entities import DetectionResult

#这个类的功能是计算二分类指标，用于评估检测器的性能。
# 它统计真阳性、假阳性、真阴性和假阴性的数量，并计算精确率、召回率、F1分数和准确率等指标。
# 它还提供了一个方法来打印这些指标的报告，以便更直观地展示评估结果。
class ClassificationMetrics:
    """
    Compute binary classification metrics for labels:
        positive_label = "malicious"
        negative_label = "normal"
    """

    def __init__(self, positive_label: str = "malicious"):
        self.positive_label = positive_label

    def evaluate(self, results: List[DetectionResult]) -> Dict:
        if not results:
            return {
                "num_samples": 0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

        tp = fp = tn = fn = 0

        for r in results:
            pred_pos = (r.predicted_label == self.positive_label)
            true_pos = (r.true_label == self.positive_label)

            if pred_pos and true_pos:
                tp += 1
            elif pred_pos and not true_pos:
                fp += 1
            elif (not pred_pos) and (not true_pos):
                tn += 1
            elif (not pred_pos) and true_pos:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(results) if results else 0.0

        return {
            "num_samples": len(results),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "accuracy": round(accuracy, 6),
        }

    def print_report(self, metrics: Dict) -> None:
        print("\n===== Classification Metrics =====")
        print(f"num_samples : {metrics['num_samples']}")
        print(f"TP          : {metrics['tp']}")
        print(f"FP          : {metrics['fp']}")
        print(f"TN          : {metrics['tn']}")
        print(f"FN          : {metrics['fn']}")
        print(f"Precision   : {metrics['precision']:.6f}")
        print(f"Recall      : {metrics['recall']:.6f}")
        print(f"F1          : {metrics['f1']:.6f}")
        print(f"Accuracy    : {metrics['accuracy']:.6f}")