import os
import pandas as pd
import numpy as np
import pickle
from datasets.time_dataset import TimeDataset
import torch
import random

SPLITS_PATH = "../../dataset/multiRedditDep/splits"
PATH_ROOT = "../../dataset/multiRedditDep/for_experiments"

class GraphRedditDataset(TimeDataset):
    def __init__(self, args, kind="train"):
        self.args = args
        self.kind = kind

        self.df = pd.read_csv(f"{SPLITS_PATH}/{kind}_users_multimodal.csv")
        self.users = self.df["user"].tolist()

        self.window_size = self.args.window_size
        self.labels = self.df["label"].tolist()

    def build_window_edges(self, available, K=2, self_loop=True):
        """
        Build temporal edges for a modality with optional self-loops and time-delta.
        available: list of 0/1 indicating if modality exists at time t
        W: window size
        self_loop: add self-loop edges
        Returns:
        edge_index: [2, num_edges] tensor
        edge_attr: [num_edges, 1] tensor (time delta)
        """
        time_ids = [i for i, a in enumerate(available) if a == 1]
        src, dst, deltas = [], [], []
        for i, t in enumerate(time_ids):
            for j, u in enumerate(time_ids):
                if i == j and self_loop:
                    src.append(i); dst.append(j); deltas.append(0)  # self-loop, delta=0
                elif i != j and abs(t - u) <= K:
                    src.append(i); dst.append(j); deltas.append(abs(t - u))
        return torch.tensor([src, dst], dtype=torch.int), torch.tensor(deltas, dtype=torch.float).view(-1, 1)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        label = self.labels[idx]
        with open(os.path.join(PATH_ROOT, self.kind, user, "timestamps.pkl"), "rb") as f:
            dates = np.array(pickle.load(f))

        image_embeddings = None
        text_embeddings = None

        ########################################
        if self.args.modality in ["image", "both"]:
            with open(
                f"{PATH_ROOT}/{self.kind}/{user}/{self.args.image_embeddings_type}.pkl",
                "rb",
            ) as f:
                image_embeddings = pickle.load(f)

        if self.args.modality in ["text", "both"]:
            with open(
                f"{PATH_ROOT}/{self.kind}/{user}/{self.args.text_embeddings_type}.pkl",
                "rb",
            ) as f:
                text_embeddings = pickle.load(f)

        ########################################
        np.random.seed()


        sample = self.load_multimodal_graph(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            label=label,
            context_size=self.args.K,
            date_threshold=self.args.date_threshold,
            dates=dates,
            user_name=user,
        )
        return sample
    