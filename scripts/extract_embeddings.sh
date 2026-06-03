#!/bin/bash
set -e

python extract_twitter_embeddings.py --modality image --embs clip
python extract_twitter_embeddings.py --modality image --embs dino

python extract_twitter_embeddings.py --modality text --embs bert
python extract_twitter_embeddings.py --modality text --embs roberta
python extract_twitter_embeddings.py --modality text --embs emoberta
python extract_twitter_embeddings.py --modality text --embs minilm
python extract_twitter_embeddings.py --modality text --embs mentalbert

python extract_twitter_embeddings.py --modality emotion --embs bert

