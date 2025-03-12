import os.path
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

class Draw():

    def __init__(self, rootdir):
        self.rootdir = self.__checkFolder(rootdir)
        self.folderpath = self.__createFoler()

    def __checkFolder(self, rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        return rootdir

    def __createFoler(self):
        folderpath = f"{self.rootdir}/pictures"
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        return folderpath

    def create_confusion_matrix(self, y_true, predictions):
        cm = sk.metrics.confusion_matrix(y_true, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.folderpath, "confusion_matrix.png"))

    def create_roc_auc(self, y_true, predictions):
        fpr, tpr, _ = sk.metrics.roc_curve(y_true, predictions)
        roc_auc = sk.metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(self.folderpath, "roc_auc.png"))

