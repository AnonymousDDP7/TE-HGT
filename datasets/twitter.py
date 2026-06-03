import glob
import json
import pickle
import numpy as np
from datasets.time_dataset import TimeDataset
import torch 

# DATA_PATH = "../../dataset/MultiModalDataset"
DATA_PATH = "../../dataset/TwitterMultiModalDataset"
EMBEDDINGS_PATH_TEXT = "./text_embedding/twitter"
EMBEDDINGS_PATH_IMAGES = "./image_embedding/twitter"

class GraphTwitterDataset(TimeDataset):
    def __init__(self, args, kind="train"):
        self.args = args
        self.kind = kind

        positive_users = sorted(glob.glob(f"{DATA_PATH}/positive/*"))
        negative_users = sorted(glob.glob(f"{DATA_PATH}/negative/*"))
        # print(f"Total positive users: {len(positive_users)}")
        users_per_fold = len(positive_users) // self.args.num_folds

        start_idx_fold = self.args.fold * users_per_fold
        end_idx_fold = (self.args.fold + 1) * users_per_fold

        if self.kind in ["valid", "test"]:
            positive_users_fold = positive_users[start_idx_fold:end_idx_fold]
            negative_users_fold = negative_users[start_idx_fold:end_idx_fold]
            self.users = positive_users_fold + negative_users_fold
            self.window_size = self.args.window_size

        if self.kind == "train":
            positive_users_fold = (
                positive_users[:start_idx_fold] + positive_users[end_idx_fold:]
            )
            negative_users_fold = (
                negative_users[:start_idx_fold] + negative_users[end_idx_fold:]
            )
            self.users = positive_users_fold + negative_users_fold

            self.window_size = self.args.window_size

        self.labels = [
            0 if "negative" in user_path else 1
            for user_path in self.users
        ]

        self.positive_users = positive_users_fold
        self.negative_users = negative_users_fold

        self.users = list(map(lambda x: x.split("\\")[-1], self.users))
        
        with open("./datasets/twitter-dates.json", "rt") as f:
            self.user_dates = json.load(f)

    def __len__(self):
        return len(self.users)

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

    def __getitem__(self, idx):
        user = self.users[idx]
        label = self.labels[idx]
        # print(f"User: {user}, label: {label}")
        label_name = "positive" if label == 1 else "negative"

        dates = np.array(self.user_dates[user])

        image_embeddings = None
        text_embeddings = None
      
        ########################################
        if self.args.modality in ["image", "both"]:
            with open(
                f"{EMBEDDINGS_PATH_IMAGES}/{label_name}/{user}/{self.args.image_embeddings_type}.pkl",
                "rb",
            ) as f:
                image_embeddings = pickle.load(f)

        if self.args.modality in ["text", "both"]:
            with open(
                f"{EMBEDDINGS_PATH_TEXT}/{label_name}/{user}/{self.args.text_embeddings_type}.pkl",
                "rb",
            ) as f:
                text_embeddings = pickle.load(f)

        # with open(
        #         f"{EMBEDDINGS_PATH_TEXT}/{label_name}/{user}/{self.args.emotion_embeddings_type}.pkl",
        #         "rb",
        #     ) as f:
        #     emotion_embeddings = pickle.load(f)

        ########################################
        # if self.kind != "train":
        #     np.random.seed(28)
        # np.random.seed()
   
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
 