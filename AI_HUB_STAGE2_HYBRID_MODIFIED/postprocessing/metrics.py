class MetricsCalculator:
    """
    Calculates classification metrics from per-video results.

    Expected labels:
        accident
        normal
    """

    def calculate_confusion_matrix(self, results):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for result in results:
            actual = result["ground_truth"].lower()
            predicted = result["prediction"].lower()

            if actual == "accident" and predicted == "accident":
                tp += 1

            elif actual == "normal" and predicted == "normal":
                tn += 1

            elif actual == "normal" and predicted == "accident":
                fp += 1

            elif actual == "accident" and predicted == "normal":
                fn += 1

        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }

    def calculate_metrics(self, results):
        confusion = self.calculate_confusion_matrix(results)

        tp = confusion["TP"]
        tn = confusion["TN"]
        fp = confusion["FP"]
        fn = confusion["FN"]

        total = tp + tn + fp + fn

        if total > 0:
            accuracy = (tp + tn) / total
        else:
            accuracy = 0.0

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )
        else:
            f1 = 0.0

        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0

        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Specificity": specificity
        }