from torch.utils.data import Dataset
import os
import random
import pandas as pd
import numpy as np
import cv2
import pickle
from transformers import BertTokenizer, BertTokenizerFast

IMAGE_PATH = [
    "../../dataset/multiRedditDep/notdepressed/images",
    "../../dataset/multiRedditDep/depressed/images",
]

SPLITS_PATH = "../../dataset/multiRedditDep/splits"
DATA_PATH = "../../dataset/multiRedditDep/splits"

class SubmissionsDataset(Dataset):
    def __init__(self, args=None, kind="train"):
        self.args = args
        self.kind = kind

        self.df = pd.read_csv(
            f"{SPLITS_PATH}/{kind}_users_multimodal.csv",
            # lineterminator="\n",
            # low_memory=False,
        )

        # self.df = self.df[self.df.user == "user3320"]
        self.users = self.df["user"].tolist()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        try:
            user_df = pd.read_csv(
                f"{DATA_PATH}/{self.kind}/{user}.csv" #, lineterminator="\n"
            )
            # print(user_df)
            label = int(user_df["label"].unique()[0])
        except:
            user_df = pd.read_csv(
                f"{DATA_PATH}/{self.kind}/{user}.csv", lineterminator="\n"
            )
            # print(user_df)
            label = int(user_df["label\r"].unique()[0])            

        user_df["title"] = user_df["title"].replace(np.nan, "")
        user_df["selftext"] = user_df["selftext"].replace(np.nan, "")
        user_df["body"] = user_df["body"].replace(np.nan, "")

        user_df["fulltext"] = (
            user_df["title"] + " " + user_df["selftext"] + " " + user_df["body"]
        )

        images = []
        for path in user_df["image_path"]:
            if not os.path.isfile(f"{IMAGE_PATH[label]}/{path}"):
                img = np.nan
            else:
                img = cv2.imread(f"{IMAGE_PATH[label]}/{path}")
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            images.append(img)

        sample = {
            "user": user,
            "ids": user_df["id"].tolist(),
            "date": user_df["created_utc"].tolist(),
            "texts": user_df["fulltext"].tolist(),
            "images_paths": user_df["image_path"].tolist(),
            "images": images,
        }

        return sample