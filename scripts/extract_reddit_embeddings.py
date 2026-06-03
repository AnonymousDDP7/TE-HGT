import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import pickle
import argparse
import numpy as np
from transformers import pipeline
from transformers import (
    ViTFeatureExtractor,
    ViTModel,
    CLIPProcessor,
    CLIPModel,
    CLIPVisionModel,
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaTokenizer,
    BertModel,
    BertTokenizer,
    AutoModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
import cv2 

def extract_content(line, avail_imgs):
    data = json.loads(line)
    # title = data.get('title') or ''
    # main = data.get('selftext') or data.get('body') or data.get('text') or ''
    # if title and main:
    #     full_text = f"{title} . {main}"
    # else:
    #     full_text = title + main

    # First available URL: direct post URL, else first preview image URL
    img_name = np.nan
    url_val = data.get('url')
    if not url_val:
        preview = data.get('preview')
        if isinstance(preview, dict):
            images = preview.get('images')
            if images:
                source = images[0].get('source')
                if isinstance(source, dict):
                    url_val = source.get('url')
        
    if url_val:
        img_name = url_val.split('?', 1)[0].rsplit('/', 1)[-1]
        if img_name not in avail_imgs:
            img_name = np.nan

    # Timestamp
    # timestamp = data.get('timestamp') or data.get('created_utc', 0.0)
    return data, img_name


def extract_image(img_path, model, input_representation, device):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Unable to read image at {img_path}. Skipping.")
        return np.nan
    
    image = image.transpose(2, 0, 1)
    image = input_representation(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**image)
        embedding = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embedding

def extract_emotion(text, model):
    with torch.no_grad():
        embedding = model(text, truncation=True, max_length=512, batch_size=32)
        encoded_texts = np.array([np.array([item['score'] for item in  embedding[i]]) for i in range(len( embedding))])
    return encoded_texts

def extract_text(text, model, device):
    with torch.no_grad():
        embedding = model.encode(text, batch_size=128, convert_to_numpy=True)
    return embedding

if __name__ == "__main__":

    PATH_ROOT = "D:/1.Postdoc/Project/Depression/dataset/multiRedditDep"
    PATH2SAVE = "D:/1.Postdoc/Project/Depression/dataset/multiRedditDep/for_experiments"


    kind = "test"  # "train", "valid", "test"
    data = pd.read_csv(os.path.join(PATH_ROOT, "splits/{}_users_multimodal.csv".format(kind)))
    list_users = list(data['user'].unique())

    list_img_depressed = os.listdir(f"{PATH_ROOT}/depressed/images")
    list_img_notdepressed = os.listdir(f"{PATH_ROOT}/notdepressed/images")
    ##### Load Model #####
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # img_representation = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    # model_img = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", ).to(device)
    # model_img = model_img.to(device)

    img_representation = ViTFeatureExtractor.from_pretrained("facebook/dino-vitb16")
    model_img = ViTModel.from_pretrained("facebook/dino-vitb16").to(device)
    model_img = model_img.to(device)

    ####Load local pretrained Mental-BERT model with absolute path
    text_encoder = SentenceTransformer("../mental-bert-base-uncased").cuda()
    ### Emotion
    # classifier = pipeline(task="text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=0)


    for user in tqdm(list_users):
        # if os.path.exists(os.path.join(PATH2SAVE, kind, user)):
        #     continue  # Skip already processed users
        if os.path.isfile(os.path.join(PATH2SAVE, kind, user, "j-hartmann.pkl")):
            continue  # Skip already processed users

        # print(f"Processing user: {user}")
        # os.makedirs(os.path.join(PATH2SAVE, kind, user), exist_ok=True)

        label = data[data['user'] == user]['label'].values[0]

        if label == 0:
            user_path = os.path.join(PATH_ROOT, "notdepressed/texts", user+".jl")
            list_imgs = list_img_notdepressed
        else:
            user_path = os.path.join(PATH_ROOT, "depressed/texts", user+".jl")
            list_imgs = list_img_depressed

        list_img_each_user = []
        lis_text_each_user = []
        list_timestamp_each_user = []
        posts_buffer = []

        with open(user_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                try:
                    # print(f"Processing post {i} for user {user}", len(list_img_each_user), len(lis_text_each_user))
                    # Extract components
                    df, img_name = extract_content(line, list_imgs)

                    title = df.get('title', "")
                    selftext = df.get('selftext', "")
                    body = df.get('body', "")

                    if pd.isna(img_name) and title + selftext + body == '':
                        continue  # Skip posts with no content
                    ##
                    # if any(post['content'] == full_text for post in posts_buffer) or "[removed]" in full_text or "[deleted]" in full_text:
                    #     # print("Duplicate or removed content found, skipping.")
                    #     continue  # Skip duplicate content
                    
                    id_link = df.get('link_id', '')
                    if id_link != '':
                        id = id_link[3:]
                    else:
                        id = df.get('id', '')

                    posts_buffer.append({
                            'user': user,
                            "id": id,
                            'timestamp': float(df.get('created_utc', 0.0)),
                            "subreddit": df.get('subreddit', ''),
                            'title': title,
                            'selftext': selftext,
                            'body': body,
                            'image': img_name,
                            'label': label,
                        })
                    

                except ValueError:
                    continue # Skip malformed lines
    
        ## Create dataframe from
        posts_buffer = pd.DataFrame(posts_buffer)
        # posts_buffer.to_csv(os.path.join(PATH2SAVE, kind, user, "data.csv"), index=False)
        # Merge title, selftext, body for each id+subreddit combination
        merged = posts_buffer.groupby(['id', 'subreddit']).agg({
            'title': lambda x: ' . '.join(x[x.notna()].unique()),
            'selftext': lambda x: ' . '.join(x[x.notna()].unique()),
            'body': lambda x: ' . '.join(x[x.notna()].unique()),
            'timestamp': "min",  # Keep all timestamps as a list
            'user': 'first',
            'image': 'first',
            'label': 'first'
        }).reset_index()
        # Combine title, selftext, body into one 'content' column
        merged['content'] = (
            merged[['title', 'selftext', 'body']]
            .fillna('')
            .agg(lambda x: ' . '.join(filter(None, x)), axis=1)
        )
        # Drop the original columns if you don't need them
        merged = merged.drop(['title', 'selftext', 'body'], axis=1)
        merged = merged[merged['content'].str.len() >= 50]  # Keep posts with content length >= 50
        merged = merged.sort_values(by="timestamp", ascending=False).reset_index(drop=True)

        # Save merged dataframe
        merged.to_csv(os.path.join(PATH2SAVE, kind, user, "data.csv"), index=False)

        merged = pd.read_csv(os.path.join(PATH2SAVE, kind, user, "data.csv"))
        ### Extract image embeddings
        for idx in range(len(merged)):
            if not pd.isna(merged["image"][idx]):
                image_embedding = extract_image(os.path.join(PATH_ROOT, f"{'notdepressed' if label==0 else 'depressed'}/images", merged["image"][idx]), 
                                        model_img, 
                                        img_representation,
                                        device)
                print(f"Image embedding shape: {image_embedding.shape}")
            else:
                image_embedding = np.nan
            list_img_each_user.append(image_embedding)

            
        ####Extract text embeddings
        with torch.no_grad():
            text_embedding = extract_text(list(merged["content"]), text_encoder, device)
            # emotion_embedding = extract_emotion(list(merged["content"]), classifier)
        # list_timestamp_each_user = list(merged["timestamp"])
        # print(f"Processing user: {user}", len(emotion_embedding))
        # print(len(list_img_each_user))
        assert len(list_timestamp_each_user) == len(list_img_each_user) == len(text_embedding), "Different in length"
            
        # Save embeddings
        with open(os.path.join(PATH2SAVE, kind, user, "dino.pkl"), 'wb') as f_img:
            pickle.dump(list_img_each_user, f_img)

        with open(os.path.join(PATH2SAVE, kind, user, "mentalbert.pkl"), 'wb') as f_text:
            pickle.dump(text_embedding, f_text)

        with open(os.path.join(PATH2SAVE, kind, user, "timestamps.pkl"), 'wb') as f_time:
            pickle.dump(list_timestamp_each_user, f_time)

        # with open(os.path.join(PATH2SAVE, kind, user, "j-hartmann.pkl"), 'wb') as f_time:
        #     pickle.dump(emotion_embedding, f_time)
