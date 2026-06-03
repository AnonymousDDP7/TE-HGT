from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import glob
import json
import tqdm
import os
import ast

DATA_PATH = [
    "D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/notdepressed",
    "D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/depressed",
]

EXPERIMENTS_DATA_PATH = "D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/splits"

files = sorted(glob.glob(DATA_PATH[0]) + glob.glob(DATA_PATH[1]))

list_img_notdepressed = os.listdir(f"{DATA_PATH[0]}/images")
list_img_depressed = os.listdir(f"{DATA_PATH[1]}/images")
list_imgs = list_img_depressed + list_img_notdepressed

# def get_users_labels(path):
#     data_df = pd.read_csv(path)
#     users = data_df["user"]
#     labels = data_df["label"]
#     for i in range(len(users)):
#         print(i, users[i], labels[i])
#     return list(users), list(labels)

# Extract image name from URL or preview
def extract_image_name(row):
    img_name = np.nan
    url_val = row.get('url')
    if not url_val:
        preview = row.get('preview')
        if isinstance(preview, dict):
            images = preview.get('images')
            if images:
                source = images[0].get('source')
                if isinstance(source, dict):
                    url_val = source.get('url')
    
    if url_val:
        img_name = url_val.split('?', 1)[0].rsplit('/', 1)[-1]
        if img_name not in list_imgs:
            img_name = np.nan
    return img_name
    
def make_dataset(path, kind="train"):
    data_df = pd.read_csv(path)
    
    # Debug: Check label distribution in input CSV
    print(f"\n=== Processing {kind} ===")
    print(f"Total users: {len(data_df)}")
    print(f"Label distribution:\n{data_df['label'].value_counts()}")

    # for user, label in tqdm.tqdm(zip(users, labels), total=len(users)):
    for idx in range(len(data_df)):
        meta_list = []
        
        label = int(data_df["label"][idx])
        user = str(data_df["user"][idx])
        # print(f"Processing: idx={idx}, user={user}, label={label}")

        # if str("user1841") == user or str("user717") == user:
        #     print(f"Processing: idx={idx}, user={user}, label={label}")
        # else:
        #     continue

        # print("Processing: ", user)

        data_path = DATA_PATH[label]
        user_file = f"{data_path}/texts/{user}.jl"

        with pd.read_json(user_file, lines=True, chunksize=50000, nrows=50000) as reader:
            for chunk in reader:
                
                if "body" not in chunk.columns:
                    chunk["body"] = np.nan

                if "selftext" not in chunk.columns:
                    chunk["selftext"] = np.nan

                if "title" not in chunk.columns:
                    chunk["title"] = np.nan

                if "preview" not in chunk.columns:
                    chunk["preview"] = np.nan


                chunk = chunk[
                    [
                        "id",
                        "subreddit",
                        "created_utc",
                        "title",
                        "selftext",
                        "body",
                        "preview",
                    ]
                ]


                chunk["image_path"] = chunk.apply(extract_image_name, axis=1)

                # print(':::: Images after checking if they exist: ', len(chunk[chunk['image_path'].notna() == True]))

                chunk["selftext"] = chunk["selftext"].apply(
                    lambda x: x if str(x) != "[removed]" else np.nan
                )

                chunk["label"] = label
                # print(label)
                # print(':::: User: ', chunk['author'].unique()[0])
                # print(':::: Images: ', len(chunk[chunk['image_path'].notna() == True]))
                # print(':::: Posts: ', len(chunk))
                # print(':::: Label: ', chunk['label'].unique()[0])

                if len(chunk) != 0:
                    meta_list.append(chunk)


        df = pd.concat(meta_list)

        if not os.path.exists(f"{EXPERIMENTS_DATA_PATH}/{kind}"):
            os.makedirs(f"{EXPERIMENTS_DATA_PATH}/{kind}")
        df.to_csv(f"{EXPERIMENTS_DATA_PATH}/{kind}/{user}.csv", index=False)
        # print(f"{EXPERIMENTS_DATA_PATH}/{kind}/{user}.csv")
        
# train_users, train_labels = get_users_labels(r"D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/splits/train_users_multimodal.csv")
# val_users, val_labels = get_users_labels(r"D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/splits/valid_users_multimodal.csv")
# test_users, test_labels = get_users_labels(r"D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/splits/test_users_multimodal.csv")

make_dataset(r"D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/splits/train_users_multimodal.csv", kind="train")
make_dataset(r"D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/splits/valid_users_multimodal.csv", kind="valid")
make_dataset(r"D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/splits/test_users_multimodal.csv", kind="test")
