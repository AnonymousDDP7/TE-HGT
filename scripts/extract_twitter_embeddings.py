import os
import sys
import tqdm
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
# from huggingface_hub import login
# login("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")  # paste your token here
# print("Login successfully!")

sys.path.insert(0, 'datasets')
from twitter_submission_dataset import TwitterSubmissionDataset

parser = argparse.ArgumentParser(description="Do stuff.")
parser.add_argument("--modality", type=str, required=True, default='text')
parser.add_argument("--embs", type=str, required=True, default = 'mentalbert')

args = parser.parse_args()

modality = args.modality
embs = args.embs

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

IMAGE_EMBEDDINGS_PATH = "image_embedding/twitter"
TEXT_EMBEDDINGS_PATH = "text_embedding/twitter"

embs_type = {
    "image": {
        "clip": {
            "model_name": "openai/clip-vit-base-patch16",
            "model_class": CLIPVisionModel,
            "input_representation": CLIPProcessor,
        },
        "dino": {
            "model_name": "facebook/dino-vitb16",
            "model_class": ViTModel,
            "input_representation": ViTFeatureExtractor,
        },
    },
    "text": {
        "bert": {
            "model_name": "bert-base-uncased",
            "model_class": BertModel,
            "input_representation": BertTokenizer,
        },
        "roberta": {
            "model_name": "sentence-transformers/stsb-roberta-base",
            "model_class": RobertaModel,
            "input_representation": RobertaTokenizer,
        },
        "minilm": {
            "model_name": "microsoft/Multilingual-MiniLM-L12-H384",
            "model_class": AutoModel,
            "input_representation": AutoTokenizer,
        },
        "emoberta": {
            "model_name": "tae898/emoberta-base",
            "model_class": AutoModel,
            "input_representation": AutoTokenizer,
        },
        "mentalbert": {
            "model_name": "mental/mental-bert-base-uncased",
            "model_class": AutoModel,
            "input_representation": AutoTokenizer,
        },
    },
}

def extract_emotion_embedding(dataset, embs, label):
    classifier = pipeline(task="text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device="cuda") #.to("cuda")

    for i in tqdm.tqdm(range(len(dataset))):

        sample = dataset[i]
        user = sample["user"].split('\\')[-1]

        path = f"{TEXT_EMBEDDINGS_PATH}/{label}/{user}"
        os.makedirs(path, exist_ok=True)

        if (os.path.exists(f"{path}/{embs}.pkl")) and not os.stat(
            f"{path}/{embs}.pkl"
        ).st_size == 0:
            continue

        encoded_texts = classifier(
            sample["texts"])
        
        encoded_texts = np.array([np.array([item['score'] for item in  encoded_texts[i]]) for i in range(len( encoded_texts))])
        # with open(f"{path}/{embs}.pkl", "wb") as f:
        #     pickle.dump(encoded_texts, f)


def extract_text_embedding(dataset, model, embs, label):

    BATCH_SIZE = 128
    text_encoder = SentenceTransformer(model).cuda()
    
    for i in tqdm.tqdm(range(len(dataset))):

        sample = dataset[i]
        user = sample["author"]

        path = f"{TEXT_EMBEDDINGS_PATH}/{label}/{user}"
        os.makedirs(path, exist_ok=True)

        # if (os.path.exists(f"{path}/{embs}.pkl")) and not os.stat(
        #     f"{path}/{embs}.pkl"
        # ).st_size == 0:
        #     continue
        

        encoded_texts = text_encoder.encode(
            sample["texts"], batch_size=BATCH_SIZE, convert_to_numpy=True
        )
        print("Len: ", len(sample["texts"]), encoded_texts.shape)
        # print(encoded_texts.shape)  ##[L x emb_size]
        # with open(f"{path}/{embs}.pkl", "wb") as f:
        #     pickle.dump(encoded_texts, f)

def extract_image_embedding(dataset, input_representation, model, embs, label):

    for i in tqdm.tqdm(range(len(dataset))):

        sample = dataset[i]
        user = sample["author"]
        path = os.path.join(IMAGE_EMBEDDINGS_PATH, label, user) #f"{IMAGE_EMBEDDINGS_PATH}/{label}/{user}"
        os.makedirs(path, exist_ok=True)

        embeddings_list = []
      
        if (os.path.exists(f"{path}/{embs}.pkl")) and not os.stat(
            f"{path}/{embs}.pkl"
        ).st_size == 0:
            # print(f"Skip {path}/{embs}.pkl")
            continue

        for id, image in zip(sample["ids"], sample["images"]):
            if image is np.nan or image.shape == (1, 1, 3):
                embedding = np.nan

            elif image is not np.nan:

                image = image.transpose(2, 0, 1)
                inputs = input_representation(images=image, return_tensors="pt").to(
                    device
                )

                with torch.no_grad():
                    out = model(**inputs)
                    embedding = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()

            embeddings_list.append(embedding)
        # with open(f"{path}/{embs}.pkl", "wb") as f:
        #     # print(f"Save to {path}/{embs}.pkl")
        #     pickle.dump(embeddings_list, f)

def get_embeddings_patches():

    print("MODALITY ", modality, "EMBS_TYPE ", embs)
    if modality == "image":
        input_representation = embs_type[modality][embs][
            "input_representation"
        ].from_pretrained(embs_type[modality][embs]["model_name"])
    
        model = embs_type[modality][embs]["model_class"].from_pretrained(
            embs_type[modality][embs]["model_name"],
            weights_only=True,
        )
        print(device)
        model = torch.nn.DataParallel(model, device_ids=[0])
        model = model.to(device)

    for label in ["positive", "negative"]:
        print(label)

        dataset = TwitterSubmissionDataset(label=label)

        if modality == "image":
            print("Image Embedding")
            extract_image_embedding(dataset, input_representation, model, embs, label)

        elif modality == "text":
            print("Text Embedding")
            extract_text_embedding(dataset, embs_type[modality][embs]["model_name"], embs, label)
        
        elif modality == "emotion":
            print("Emotion Embedding")
            extract_emotion_embedding(dataset, embs, label)
            
get_embeddings_patches()
