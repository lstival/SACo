#author: L. Stival

import torch
import sklearn.metrics as skm

class Eval():
    def f1_score(self, y_true, y_pred):
        """
        Calculates the F1 score.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: F1 score.
        """
        return skm.f1_score(y_true, y_pred, average='binary', zero_division=0.)
    
    def precision_score(self, y_true, y_pred):
        """
        Calculates the precision score.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Precision score.
        """
        return skm.precision_score(y_true, y_pred, average='binary', zero_division=0.)
    
    def recall_score(self, y_true, y_pred):
        """
        Calculates the recall score.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Recall score.
        """
        return skm.recall_score(y_true, y_pred, average='binary', zero_division=0.)
    
    def accuracy_score(self, y_true, y_pred):
        """
        Calculates the accuracy score.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Accuracy score.
        """
        return skm.accuracy_score(y_true, y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        """
        Calculates the confusion matrix.

        Args:
            y_true (torch.Tensor): True labels.
            y_pred (torch.Tensor): Predicted labels.

        Returns:
            torch.Tensor: Confusion matrix.
        """
        return skm.confusion_matrix(y_true, y_pred)
        
    def evaluate(self, y_true, y_pred):
        f1 = self.f1_score(y_true, y_pred)
        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)
        
        return y_pred, f1, precision, recall
    
if __name__ == '__main__':

    y_true = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])

    eval = Eval()
    out, f1, precision, recall = eval.evaluate(y_true, y_pred)

    print(f"F1: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
