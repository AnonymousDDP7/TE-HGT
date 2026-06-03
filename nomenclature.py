import torch
from models import *
from datasets import *
from evaluators import MultimodalEvaluator

device = torch.device("cuda")

DATASETS = {
    "reddit": RedditDataset, # to be changed to RedditDataset, GraphRedditDataset is temporary
    "twitter": GraphTwitterDataset,  # to be changed to TwitterDataset, GraphTwitterDataset is temporary
}

EVALUATORS = {
    "multimodal-evaluator": MultimodalEvaluator,
}

MODELS = {
    "TEHGT": TEHGT,
}
