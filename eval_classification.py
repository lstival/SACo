from data_reader.EuroSAT_ms_reader import MSReadData
from classification import EvalClassification
import os

if __name__ == "__main__":
    print("Main")
    root_path = "./data/EuroSAT_test"
    batch_size = 1
    image_size = 64
    train = False
    num_classes = 10

    # Example usage
    dataset = MSReadData()
        
    dataloader = dataset.create_dataLoader(root_path, image_size, batch_size, train=train, num_workers=1)	

    # Eval to load model
    name = "20240422_053009"

    eval = EvalClassification(model_name=name)
    acc, all_outs, all_labels = eval.evaluate(dataloader)
    print(f"Accuracy: {acc}")

    import pandas as pd

    df = pd.DataFrame(columns=["label", "predicted"])
    for i, (out, label) in enumerate(zip(all_outs, all_labels)):
        predicted = out.argmax(dim=1).cpu().numpy()[0]
        label = label.cpu().numpy()[0]
        df.loc[i] = [label, predicted]

    path = f"./results/classification/{name}/"
    os.makedirs(path, exist_ok=True)
    df.to_csv(f"{path}results.csv")

    # Count the number of correct predictions per class and the % of correct predictions

    classes = dataset.dataset.labels_list
    # classes = os.listdir(path)

    df = df.astype(int)
    df["correct"] = df["label"] == df["predicted"]
    df_correct = df.groupby("label").sum()
    df_total = df.groupby("label").count()
    df_correct["total"] = df_total["predicted"]
    df_correct["predicted"] = df_correct["predicted"]
    df_correct["percentage"] = df_correct["correct"] / df_correct["total"]

    df_correct["correct"].sum() / df_correct["total"].sum()

    # Put the class name considering the order of the classes
    df_correct["class"] = classes

    df_correct.to_csv(f"{path}results_per_class.csv")
    print(df_correct.sort_values(by="percentage"))

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix

    y_true = df["label"]
    y_pred = df["predicted"]

    cm = confusion_matrix(y_true, y_pred)

    df_confusion_matrix = pd.DataFrame(cm, columns=classes, index=classes)

    # Convert absolute values to percentages per class
    df_confusion_matrix = df_confusion_matrix.div(df_confusion_matrix.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=df_confusion_matrix.values, display_labels=classes)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=90)  # Rotate x labels by 90 degrees
    plt.show()

    forest_sum = df_confusion_matrix.loc["Forest"].sum()
    print(f"Sum of values for 'Forest': {forest_sum}")
