import torch
import numpy as np
import nomenclature
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn import metrics
from evaluators import BaseEvaluator
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader  # not torch.utils.data

class MultimodalEvaluator(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.num_runs = 1
        self.dataset = nomenclature.DATASETS[self.args.dataset]
        if "reddit" in self.args.dataset:
            self.test_dataset = self.dataset(args=args, kind="test")
        else:
            self.test_dataset = self.dataset(args=args, kind="valid")

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=False,
        )

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save=False)
        return results[-1]["f1"]

    def evaluate(self, save=True):
        y_preds = []
        y_preds_proba = []
        true_labels = []

        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(self.test_dataloader, total=len(self.test_dataloader))
            ):
                for key, value in batch.items():
                    # print(key, value.shape)
                    batch[key] = value.to(nomenclature.device)

                # if  batch["image_mask"].sum() != 0:
                #     continue
                # if batch["text_mask"].sum().item() <= 64: # 0 < batch["image_mask"].sum().item() / batch["text_mask"].sum().item()  <= 0.1:
                #     pass
                # else:
                #     continue
                # print(batch["image_mask"].sum().item(), batch["text_mask"].sum().item())

                output = self.model(batch)["probas"].detach().cpu().numpy().ravel()

                labels = batch["label"].detach().cpu().numpy().ravel()

                y_preds.extend(np.round(output))
                y_preds_proba.extend(output)
                true_labels.extend(labels)

        y_preds = np.array(y_preds)
        y_preds_proba = np.array(y_preds_proba)
        true_labels = np.array(true_labels)

        fpr, tpr, thresholds = metrics.roc_curve(true_labels, y_preds_proba, pos_label=1)
        acc = metrics.accuracy_score(true_labels, y_preds)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(true_labels, y_preds)
        recall = metrics.recall_score(true_labels, y_preds)
        f1 = metrics.f1_score(true_labels, y_preds)

        cm = metrics.confusion_matrix(true_labels, y_preds)

        results = pd.DataFrame([{
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "auc": auc,
                "accuracy": acc,
                "confusion_matrix": cm
            }]
            
        )

        return results

    def evaluate_graph(self, save=True):
        self.model.eval()
        torch.set_grad_enabled(False)

        all_preds, all_probas, all_labels = [], [], []

        for i, batch in enumerate(tqdm(self.test_dataloader, total=len(self.test_dataloader))):
            batch = batch.to(nomenclature.device)
            # print(batch["text"]["x"].shape, batch["image"]["x"].shape)
            if 128 <= batch["text"]["x"].shape[0]: #0 < batch["image"]["x"].shape[0] / batch["text"]["x"].shape[0]  <= 0.2:
                pass
            else:
                continue
            # if batch["image"]["x"].shape[0] != 0:
            #     continue

            print(batch["text"]["x"].shape, batch["image"]["x"].shape)

            output = self.model(batch)["probas"].detach().cpu().numpy().ravel()
            labels = batch["label"].detach().cpu().numpy().ravel()

            all_preds.extend(np.round(output))
            all_probas.extend(output)
            all_labels.extend(labels)

        # Now compute metrics
        all_preds = np.array(all_preds)
        all_probas = np.array(all_probas)
        all_labels = np.array(all_labels)

        fpr, tpr, _ = metrics.roc_curve(all_labels, all_probas, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = metrics.accuracy_score(all_labels, all_preds)
        precision = metrics.precision_score(all_labels, all_preds)
        recall = metrics.recall_score(all_labels, all_preds)
        f1 = metrics.f1_score(all_labels, all_preds)

        cm = metrics.confusion_matrix(all_labels, all_preds)

        results = pd.DataFrame([{
            "f1": f1, "recall": recall, "precision": precision,
            "auc": auc, "accuracy": acc,
            "confusion_matrix": cm
        }])
        
        # if save:
        #     results.to_csv(
        #         f"results/{self.args.output_dir}/{self.args.group}-{self.args.name}.csv",
        #         index=False,
        #     )

        return results
