"""
Script to measure model metrics:
- Number of parameters
- FLOPs (Floating Point Operations)
- Memory consumption (peak and average)
- Inference time

Usage:
    python measure_model_metrics.py --model T-M2S --dataset twitter --window_size 128
"""

import nomenclature
import torch
import torch.nn as nn
import argparse
import time
import json
import os
from pathlib import Path
import numpy as np
import random
from utils import load_args

from torch_geometric.data import HeteroData
import torch 
from datasets.time_dataset import TimeDataset

def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters broken down by layer type.
    
    Returns:
        dict: Contains total parameters and breakdown by layer type
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    param_breakdown = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        else:
            non_trainable_params += num_params
        
        # Get layer type
        layer_type = name.split('.')[0] if '.' in name else 'root'
        if layer_type not in param_breakdown:
            param_breakdown[layer_type] = 0
        param_breakdown[layer_type] += num_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'breakdown': param_breakdown
    }

def measure_inference_time(
    model: nn.Module,
    sample_input: dict,
    device: str = 'cpu',
    num_runs: int = 100,
    warmup_runs: int = 10
) -> dict:
    """
    Measure inference time with warmup and multiple runs.
    
    Args:
        model: PyTorch model
        sample_input: Sample input batch
        device: Device to run on
        num_runs: Number of inference runs to average
        warmup_runs: Number of warmup runs
        
    Returns:
        dict: Timing statistics in milliseconds
    """
    model.eval()
    model = model.to(device)
    
    # Move sample input to device
    if isinstance(sample_input, dict):
        for key in sample_input:
            if torch.is_tensor(sample_input[key]):
                sample_input[key] = sample_input[key].to(device)
    elif isinstance(sample_input, HeteroData):
        sample_input = sample_input.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
    
    if device != 'cpu':
        torch.cuda.synchronize(device)
    
    # Actual runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device != 'cpu':
                torch.cuda.synchronize(device)
            
            start = time.time()
            _ = model(sample_input)
            
            if device != 'cpu':
                torch.cuda.synchronize(device)
            end = time.time()
            
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
    }

def build_window_edges_vectorized(mask, context_size, self_loop=True):
    """
    Optimized version: Builds edges based on sequence index proximity (temporal).
    mask: boolean array (T_win,) indicating valid steps.
    """
    # 1. Get the original indices where data exists (e.g., [0, 1, 4, 5...])
    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

    # 2. Compute pairwise distance matrix of the ORIGINAL indices
    # shape: (Num_Valid, Num_Valid)
    # diff[i, j] = original_index[j] - original_index[i]
    idx_diff = valid_indices[None, :] - valid_indices[:, None]
    abs_diff = np.abs(idx_diff)

    # 3. Boolean mask for valid edges (Distance <= K)
    edge_mask = abs_diff <= context_size
    
    if not self_loop:
        np.fill_diagonal(edge_mask, False)

    # 4. Convert to Sparse COO format
    # src, dst are indices into 'valid_indices', which map exactly to Graph Node Indices (0..M)
    src, dst = np.nonzero(edge_mask)
    attr = abs_diff[src, dst]

    edge_index = torch.stack([torch.from_numpy(src), torch.from_numpy(dst)]).long()
    edge_attr = torch.from_numpy(attr).float().view(-1, 1)

    return edge_index, edge_attr

def load_multimodal_graph_v3(
    context_size,
    W_size,
    aval_rate,
):
    """
    Builds a hetero-graph with node types: text, emotion, image
    and relations: temporal (within each type) and aligns_with (cross-modal at same time step).

    Assumptions:
    - text/emotion exist for every time step (prior to padding).
    - images may be missing at some time steps (mask = False).
    - self.build_window_edges(mask, W, self_loop=True) maps from timeline to compact indices.
    """
    text_embeddings = np.ones((W_size, 768), dtype=np.float32)
    image_embeddings = np.ones((W_size, 768), dtype=np.float32)
    # ---------- basic prep ----------
    dates = np.arange(W_size).reshape(-1)
    random.shuffle(dates)
    dates = dates[:int(W_size * aval_rate)]
    N = len(dates)



    order_idx = np.argsort(dates).ravel()

    high_ = N - W_size
    if high_ <= 0:
        start_idx = 0
        end_idx = N
        padding_amount = -high_  # add this many "future" padded steps            


    idxs = np.asarray(order_idx[start_idx:end_idx], dtype=np.int64)

    # Slice arrays to the window
    t_emb = np.asarray(text_embeddings, dtype=np.float32)[idxs]
    d_sel = dates[idxs]

    # ---------- image embeddings + mask ----------
    T_win = len(idxs)

    img_mat = np.zeros((T_win, 768), dtype=np.float32)
    img_mask = np.zeros(T_win, dtype=bool)

    # image_embeddings can be a list (with possible floats/None) or an array
    for i, orig_idx in enumerate(idxs):
        item = image_embeddings[orig_idx]
        # treat missing/invalid entries as no-image
        if item is None or isinstance(item, (float, int, np.floating)):
            continue
        arr = np.asarray(item)
        if arr.ndim == 1:
            arr = arr.astype(np.float32, copy=False)
            if arr.size == 768 and np.all(np.isfinite(arr)):
                img_mat[i] = arr
                img_mask[i] = True
        else:
            flat = arr.reshape(-1).astype(np.float32, copy=False)
            if flat.size == 768 and np.all(np.isfinite(flat)):
                img_mat[i] = flat
                img_mask[i] = True
            # else: leave as zeros (mask remains False)

    # ---------- timestamps normalization (if needed) ----------
    # stored but not directly used below (build_window_edges may use internal state)


    # ---------- padding to window_size (if any) ----------
    # text/emotion exist for each chosen step; image may be missing
    txt_mask = np.ones(T_win, dtype=bool)

    # ---------- build graph ----------
    data = HeteroData()

    # Node features (only valid entries; padded steps are dropped by mask)
    data['text'].x    = torch.from_numpy(t_emb[txt_mask]).float()
    data['image'].x   = torch.from_numpy(img_mat[img_mask]).float()


    edge_index_txt, edge_attr_txt = build_window_edges_vectorized(txt_mask, context_size, self_loop=True)
    edge_index_img, edge_attr_img = build_window_edges_vectorized(img_mask, context_size, self_loop=True)

    data['text',   'temporal', 'text'  ].edge_index = edge_index_txt
    data['text',   'temporal', 'text'  ].edge_attr  = edge_attr_txt
    data['image',  'temporal', 'image' ].edge_index = edge_index_img
    data['image',  'temporal', 'image' ].edge_attr  = edge_attr_img

    # ---------- cross-modal edges (same timepoint) ----------
    # Build timeline -> compact index maps for each type
    def mask_to_index(mask: np.ndarray) -> np.ndarray:
        out = np.full(mask.shape[0], -1, dtype=np.int64)
        out[mask] = np.arange(int(mask.sum()), dtype=np.int64)
        return out

    txt_map = mask_to_index(txt_mask)
    img_map = mask_to_index(img_mask)

    # ---------- directional temporal relations: past and future ----------
    # We split temporal edges into two directed types for both text and image.
    # Past edges: from earlier timestep to later timestep (messages flow from past to current).
    # Future edges: from later timestep to earlier timestep (messages flow from future to current).
    date_threshold = 30
    
    # # Text past/future
    txt_positions = np.where(txt_mask)[0].astype(np.int64)
    past_src_t, past_dst_t, past_dt_t = [], [], []
    fut_src_t, fut_dst_t, fut_dt_t = [], [], []
    for i_idx in range(len(txt_positions)):
        i_pos = int(txt_positions[i_idx])
        for j_idx in range(i_idx + 1, len(txt_positions)):
            j_pos = int(txt_positions[j_idx])
            d = float(txt_positions[j_idx] - txt_positions[i_idx])
            if d <= date_threshold:
                si = int(txt_map[i_pos]); sj = int(txt_map[j_pos])
                past_src_t.append(si); past_dst_t.append(sj); past_dt_t.append(d)
                fut_src_t.append(sj);  fut_dst_t.append(si);  fut_dt_t.append(d)
            else:
                break
    if len(past_src_t) > 0:
        data['text', 'past', 'text'].edge_index = torch.tensor([past_src_t, past_dst_t], dtype=torch.long)
        data['text', 'past', 'text'].edge_attr  = torch.tensor(past_dt_t, dtype=torch.float).view(-1, 1)
        data['text', 'future', 'text'].edge_index = torch.tensor([fut_src_t, fut_dst_t], dtype=torch.long)
        data['text', 'future', 'text'].edge_attr  = torch.tensor(fut_dt_t, dtype=torch.float).view(-1, 1)
    else:
        data['text', 'past', 'text'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['text', 'past', 'text'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)
        data['text', 'future', 'text'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['text', 'future', 'text'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)

    # Image past/future (only on available image steps)
    img_positions = np.where(img_mask)[0].astype(np.int64)
    past_src_i, past_dst_i, past_dt_i = [], [], []
    fut_src_i, fut_dst_i, fut_dt_i = [], [], []
    for i_idx in range(len(img_positions)):
        i_pos = int(img_positions[i_idx])
        for j_idx in range(i_idx + 1, len(img_positions)):
            j_pos = int(img_positions[j_idx])
            d = float(img_positions[j_idx] - img_positions[i_idx])
            if d <= date_threshold:
                si = int(img_map[i_pos]); sj = int(img_map[j_pos])
                past_src_i.append(si); past_dst_i.append(sj); past_dt_i.append(d)
                fut_src_i.append(sj);  fut_dst_i.append(si);  fut_dt_i.append(d)
            else:
                break
            
    if len(past_src_i) > 0:
        data['image', 'past', 'image'].edge_index = torch.tensor([past_src_i, past_dst_i], dtype=torch.long)
        data['image', 'past', 'image'].edge_attr  = torch.tensor(past_dt_i, dtype=torch.float).view(-1, 1)
        data['image', 'future', 'image'].edge_index = torch.tensor([fut_src_i, fut_dst_i], dtype=torch.long)
        data['image', 'future', 'image'].edge_attr  = torch.tensor(fut_dt_i, dtype=torch.float).view(-1, 1)
    else:
        data['image', 'past', 'image'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['image', 'past', 'image'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)
        data['image', 'future', 'image'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['image', 'future', 'image'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)

    # text <-> image at timesteps where both exist
    both_t = np.where(txt_mask & img_mask)[0]
    ti_src = torch.from_numpy(txt_map[both_t]).long()
    ti_dst = torch.from_numpy(img_map[both_t]).long()
    zeros_ti = torch.zeros((ti_src.numel(), 1), dtype=torch.float32)

    data['text',  'aligns_with', 'image'].edge_index = torch.stack([ti_src, ti_dst], dim=0)
    data['text',  'aligns_with', 'image'].edge_attr  = zeros_ti
    data['image', 'aligns_with', 'text' ].edge_index = torch.stack([ti_dst, ti_src], dim=0)
    data['image', 'aligns_with', 'text' ].edge_attr  = zeros_ti
    
    print(t_emb.shape, img_mat.shape, len(past_src_i))
    return data

class GraphTwitterDataset(TimeDataset):
    def __init__(self, args):
        self.args = args
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = load_multimodal_graph_v3(context_size=args.K, W_size=args.window_size, aval_rate=self.args.aval_rate)

        return sample
    

from torch_geometric.loader import DataLoader  # not torch.utils.data


parser = argparse.ArgumentParser(description="Do stuff.")

parser.add_argument("--config_file", type=str, default="./configs/base_config_graph.yaml")
parser.add_argument("--model", type=str, default="HGNNv8")
parser.add_argument("--dataset", type=str, default="twitter")
parser.add_argument("--window_size", type=int, default=1024)
parser.add_argument("--position_embeddings", type=str, default="time2vec")
parser.add_argument("--image_embeddings_type", type=str, default="clip")
parser.add_argument("--text_embeddings_type", type=str, default="mentalbert")
parser.add_argument("--emotion_embeddings_type", type=str, default="j-hartmann")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--K", type=int, default=20)
parser.add_argument("--aval_rate", type=int, default=0.1)


args = parser.parse_args()
args, cfg = load_args(args)
if __name__ == '__main__':
    model = nomenclature.MODELS[args.model](args)
    model.eval()
    model.train(False)
    model.cuda()

    num_params = count_parameters(model)
    print("::: Model parameters:", num_params["total"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = GraphTwitterDataset(args=args)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
    )
    for i, sample in enumerate(test_dataloader):
        sample = sample.to(device)
        break  # just one sample for measurement

    out = measure_inference_time(model, sample, device, num_runs=100, warmup_runs=10)  # Warmup
    print(out)