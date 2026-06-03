from turtle import forward

import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import torch 

class TimeDataset(Dataset):
    def __len__(self):
        return len(self.users)

    def _emb_size(self, kind):
        if kind == "image":
            return self.args.IMAGE_EMBEDDING_SIZES[self.args.image_embeddings_type]

        if kind == "text":
            return self.args.TEXT_EMBEDDING_SIZES[self.args.text_embeddings_type]

    def build_window_edges_vectorized(self, mask, context_size, self_loop=True):
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

    def build_long_range_edges(self, mask, timestamps, context_size):
        """
        Optimized version: Handles >2 nodes but 0 edges (Sparse Timeline).
        """
        # 0. Safety casts
        mask = np.asarray(mask, dtype=bool)
        timestamps = np.asarray(timestamps)

        # 1. Extract timestamps for valid nodes
        # print(len(timestamps), len(mask))
        t_valid = timestamps[mask]
        
        # CASE A: Not enough nodes to form an edge
        if len(t_valid) < 2:
            return self._empty_edges()

        # 2. Compute difference matrix
        diff_mat = t_valid[None, :] - t_valid[:, None]

        # 3. Create Mask: Strictly Future AND Within Context
        # triu(k=1) ensures i < j (past -> future direction)
        triu_mask = np.triu(np.ones_like(diff_mat, dtype=bool), k=1)
        valid_mask = triu_mask & (diff_mat <= context_size)
        
        # 4. Get indices
        src, dst = np.nonzero(valid_mask)
        
        # CASE B: Nodes exist, but they are too far apart (No edges found)
        if len(src) == 0:
            return self._empty_edges()

        # 5. Extract attributes for valid edges only
        deltas = diff_mat[src, dst]

        # 6. Tensor conversion
        src_t = torch.from_numpy(src).long()
        dst_t = torch.from_numpy(dst).long()
        delta_t = torch.from_numpy(deltas).float().view(-1, 1)

        # PAST: src(earlier) -> dst(later)
        past_index = torch.stack([src_t, dst_t])
        
        # FUTURE: dst(later) -> src(earlier)
        future_index = torch.stack([dst_t, src_t])
        
        return past_index, delta_t, future_index, delta_t

    def _empty_edges(self):
        """Helper to return consistent empty edge formats."""
        empty_idx = torch.zeros((2, 0), dtype=torch.long)
        empty_attr = torch.zeros((0, 1), dtype=torch.float)
        # Returns: past_idx, past_attr, fut_idx, fut_attr
        return empty_idx, empty_attr, empty_idx, empty_attr

    
    def load_multimodal_graph(
        self,
        image_embeddings,        # list/array: each item is np.ndarray[D_img] or a sentinel (e.g., float/None)
        text_embeddings,         # np.ndarray[T, D_txt]
        dates,                   # array-like[T], seconds since epoch or similar
        label,
        context_size,
        date_threshold,
        user_name=None
    ):
        """
        Builds a hetero-graph with node types: text, emotion, image
        and relations: temporal (within each type) and aligns_with (cross-modal at same time step).

        Assumptions:
        - text/emotion exist for every time step (prior to padding).
        - images may be missing at some time steps (mask = False).
        - self.build_window_edges(mask, W, self_loop=True) maps from timeline to compact indices.
        """

        # ---------- basic prep ----------
        dates = np.asarray(dates, dtype=np.float64).reshape(-1)
        N = len(dates)
        assert N == len(text_embeddings), \
            "dates, text_embeddings must have the same length"

        # Determine ordering over time
        if getattr(self.args, "position_embeddings", None) != "zero":
            order_idx = np.argsort(dates).ravel()
        else:
            order_idx = np.random.permutation(N)

        # Window selection (pad if window is larger than sequence length)
        np.random.seed()
        W_size = int(getattr(self, "window_size", N))
        high_ = N - W_size
        if high_ <= 0:
            start_idx = 0
            end_idx = N
            padding_amount = -high_  # add this many "future" padded steps            
        elif self.kind in ["valid"] : ##if N > W_size
            start_idx = N - W_size 
            end_idx = N 
            padding_amount = 0  # add this many "future" padded steps
        else:
            # include the last valid start by using high_ + 1
            start_idx = np.random.randint(0, high_ + 1)
            end_idx = start_idx + W_size
            padding_amount = 0


        idxs = np.asarray(order_idx[start_idx:end_idx], dtype=np.int64)

        # Slice arrays to the window
        t_emb = np.asarray(text_embeddings, dtype=np.float32)[idxs]
        d_sel = dates[idxs]

        # ---------- image embeddings + mask ----------
        img_dim = int(self._emb_size(kind="image"))
        T_win = len(idxs)

        img_mat = np.zeros((T_win, img_dim), dtype=np.float32)
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
                if arr.size == img_dim and np.all(np.isfinite(arr)):
                    img_mat[i] = arr
                    img_mask[i] = True
            else:
                flat = arr.reshape(-1).astype(np.float32, copy=False)
                if flat.size == img_dim and np.all(np.isfinite(flat)):
                    img_mat[i] = flat
                    img_mask[i] = True
                # else: leave as zeros (mask remains False)

        # ---------- timestamps normalization (if needed) ----------
        # stored but not directly used below (build_window_edges may use internal state)
        if getattr(self.args, "timestamp_kind", None) == "delta":
            d_proc = np.hstack(([0.0], np.diff(d_sel))) / 3600.0  # hours
        elif getattr(self.args, "timestamp_kind", None) == "relative":
            d_proc = (d_sel - np.min(d_sel)) / 3600.0
        else:
            d_proc = d_sel.astype(np.float64)
            
        dates = dates / 24. # Date difference

        # ---------- padding to window_size (if any) ----------
        # text/emotion exist for each chosen step; image may be missing
        txt_mask = np.ones(T_win, dtype=bool)

        # ---------- build graph ----------
        data = HeteroData()

        # Node features (only valid entries; padded steps are dropped by mask)
        data['text'].x    = torch.from_numpy(t_emb[txt_mask]).float()
        data['image'].x   = torch.from_numpy(img_mat[img_mask]).float()


        edge_index_txt, edge_attr_txt = self.build_window_edges_vectorized(txt_mask, context_size, self_loop=True)
        edge_index_img, edge_attr_img = self.build_window_edges_vectorized(img_mask, context_size, self_loop=True)

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
        # date_threshold = 30
        
        # # Text past/future
        txt_positions = np.where(txt_mask)[0].astype(np.int64)
        back_src_t, back_dst_t, back_dt_t = [], [], []
        forward_src_t, forward_dst_t, forward_dt_t = [], [], []
        for i_idx in range(len(txt_positions)):
            i_pos = int(txt_positions[i_idx])
            for j_idx in range(i_idx + 1, len(txt_positions)):
                j_pos = int(txt_positions[j_idx])
                d = float(d_proc[j_idx] - d_proc[i_idx])
                if d <= date_threshold:
                    si = int(txt_map[i_pos]); sj = int(txt_map[j_pos])
                    back_src_t.append(si); back_dst_t.append(sj); back_dt_t.append(d)
                    forward_src_t.append(sj);  forward_dst_t.append(si);  forward_dt_t.append(d)
                else:
                    break
        if len(back_src_t) > 0:
            data['text', 'back', 'text'].edge_index = torch.tensor([back_src_t, back_dst_t], dtype=torch.long)
            data['text', 'back', 'text'].edge_attr  = torch.tensor(back_dt_t, dtype=torch.float).view(-1, 1)
            data['text', 'forward', 'text'].edge_index = torch.tensor([forward_src_t, forward_dst_t], dtype=torch.long)
            data['text', 'forward', 'text'].edge_attr  = torch.tensor(forward_dt_t, dtype=torch.float).view(-1, 1)
        else:
            data['text', 'back', 'text'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data['text', 'back', 'text'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)
            data['text', 'forward', 'text'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data['text', 'forward', 'text'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)

        # Image past/future (only on available image steps)
        img_positions = np.where(img_mask)[0].astype(np.int64)
        back_src_i, back_dst_i, back_dt_i = [], [], []
        forward_src_i, forward_dst_i, forward_dt_i = [], [], []
        for i_idx in range(len(img_positions)):
            i_pos = int(img_positions[i_idx])
            for j_idx in range(i_idx + 1, len(img_positions)):
                j_pos = int(img_positions[j_idx])
                d = float(d_proc[j_idx] - d_proc[i_idx])
                if d <= date_threshold:
                    si = int(img_map[i_pos]); sj = int(img_map[j_pos])
                    back_src_i.append(si); back_dst_i.append(sj); back_dt_i.append(d)
                    forward_src_i.append(sj);  forward_dst_i.append(si);  forward_dt_i.append(d)
                else:
                    break
                
        if len(back_src_i) > 0:
            data['image', 'back', 'image'].edge_index = torch.tensor([back_src_i, back_dst_i], dtype=torch.long)
            data['image', 'back', 'image'].edge_attr  = torch.tensor(back_dt_i, dtype=torch.float).view(-1, 1)
            data['image', 'forward', 'image'].edge_index = torch.tensor([forward_src_i, forward_dst_i], dtype=torch.long)
            data['image', 'forward', 'image'].edge_attr  = torch.tensor(forward_dt_i, dtype=torch.float).view(-1, 1)
        else:
            data['image', 'back', 'image'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data['image', 'back', 'image'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)
            data['image', 'forward', 'image'].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data['image', 'forward', 'image'].edge_attr  = torch.zeros((0, 1), dtype=torch.float)

        # text <-> image at timesteps where both exist
        both_t = np.where(txt_mask & img_mask)[0]
        ti_src = torch.from_numpy(txt_map[both_t]).long()
        ti_dst = torch.from_numpy(img_map[both_t]).long()
        zeros_ti = torch.zeros((ti_src.numel(), 1), dtype=torch.float32)

        data['text',  'aligns_with', 'image'].edge_index = torch.stack([ti_src, ti_dst], dim=0)
        data['text',  'aligns_with', 'image'].edge_attr  = zeros_ti
        data['image', 'aligns_with', 'text' ].edge_index = torch.stack([ti_dst, ti_src], dim=0)
        data['image', 'aligns_with', 'text' ].edge_attr  = zeros_ti

        # Label
        data["label"] = torch.tensor([[label]], dtype=torch.float32)

        return data
    
   