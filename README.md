# Robust Temporal Edge-aware Heterogenous Graph for Social Media Depression Detection with Incomplete Modalities


## 🍓 Dataset Preparation

The Twitter dataset could be downloaded [here](https://drive.google.com/open?id=11ye00sHFY5re2NOBRKreg-tVbDNrc7Xd).

Please contact the author in below referenced paper for accessing the Reddit dataset.

Uban, Ana-Sabina, Berta Chulvi, and Paolo Rosso. [Explainability of Depression Detection on Social Media: From Deep Learning Models to Psychological Interpretations and Multimodality](https://link.springer.com/chapter/10.1007/978-3-031-04431-1_13). In Early Detection of Mental Health Disorders by Social Media Monitoring, pp. 289-320. Springer, Cham, 2022.

```python
# Twitter
python extract_twitter_embeddings.py --modality image --embs dino
python extract_twitter_embeddings.py --modality text --embs mentalbert
```

## 🚀 Training and Evaluating
```python

python main_graph_twitter.py --dataset twitter --window_size 128 --K 30 --group TEGHT --fold 0

python evaluate_graph_twitter.py --name checkpoint_folder --group TEGHT --fold 0 --window_size 128

