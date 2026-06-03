from torch_geometric.nn import  global_max_pool
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_add

# ---------- utilities ----------
class TimeEncoder(nn.Module):
    """Sinusoidal time encoding for scalar delta_t -> vector."""
    def __init__(self, dim=16):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, dt):
        # dt: [E,1] or [E] (scalar)
        x = dt.view(-1, 1) * self.inv_freq.view(1, -1)  # -> [E, dim/2]
        sin = torch.sin(x)
        cos = torch.cos(x)
        return torch.cat([sin, cos], dim=-1)  # [E, dim]


# ---------- relation-level edge-aware attention ----------
class EdgeAwareRelationalAttention(nn.Module):
    """
    Relation-level multi-head attention that uses edge_attr embeddings.
    Processes one relation (src_type, rel, dst_type) at a time.
    """
    def __init__(self, in_dim, hidden_dim, heads=4, time_emb_dim=16, dropout=0.1):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim ** -0.5

        # linear projections
        self.lin_k = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin_v = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin_q = nn.Linear(in_dim, hidden_dim, bias=False)

        # project time encoding into per-head additive vector for keys
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.out_lin = nn.Linear(hidden_dim, in_dim)  # residual-space
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_src, x_dst, edge_index, edge_time_enc):
        """
        x_src: [N_src, in_dim]
        x_dst: [N_dst, in_dim]
        edge_index: [2, E] (Long)
        edge_time_enc: [E, time_emb_dim] or None
        returns: aggregated messages for dst nodes: [N_dst, in_dim]
        """
        device = x_src.device
        if edge_index is None or edge_index.size(1) == 0:
            # no edges: return zeros
            return torch.zeros((x_dst.size(0), x_dst.size(1)), device=device)

        src_idx = edge_index[0].long()
        dst_idx = edge_index[1].long()

        # project
        K = self.lin_k(x_src).view(-1, self.heads, self.head_dim)  # [N_src, H, d]
        V = self.lin_v(x_src).view(-1, self.heads, self.head_dim)
        Q = self.lin_q(x_dst).view(-1, self.heads, self.head_dim)  # [N_dst, H, d]

        # gather per-edge
        K_e = K[src_idx]        # [E, H, d]
        V_e = V[src_idx]        # [E, H, d]
        Q_e = Q[dst_idx]        # [E, H, d]
        # print("Edge aware attention: ", K_e.shape, V_e.shape, Q_e.shape) # H =8, d = 128/8=16
        if edge_time_enc is not None:
            # project to same heads shape: [E, H, d]
            E_proj = self.time_mlp(edge_time_enc).view(-1, self.heads, self.head_dim)
        else:
            E_proj = torch.zeros_like(K_e)
        # print(" edge_time_enc: ", edge_time_enc.shape, E_proj.shape) 
        # modified key = K + E_proj
        K_e = K_e + E_proj

        # attention score
        scores = (Q_e * K_e).sum(dim=-1) * self.scale  # [E, H]

        # compute softmax per dst node for each head
        # alpha = []
        # for h in range(self.heads):
        #     s_h = scores[:, h]  # [E]
        #     a_h = scatter_softmax(s_h, dst_idx)  # [E]
        #     alpha.append(a_h)
        # alpha = torch.stack(alpha, dim=1)  # [E, H]
        # compute softmax per dst node for all heads at once
        alpha = scatter_softmax(scores, dst_idx, dim=0)  # [E, H]

        # weighted values
        out_edges = V_e * alpha.unsqueeze(-1)  # [E, H, d]

        # # aggregate to dst nodes
        # H_msgs = []
        # N_dst = x_dst.size(0)
        # for h in range(self.heads):
        #     m_h = out_edges[:, h, :]  # [E, d]
        #     agg_h = scatter_add(m_h, dst_idx, dim=0, dim_size=N_dst)  # [N_dst, d]
        #     H_msgs.append(agg_h)
        # # concat heads -> [N_dst, H*d] == hidden_dim
        # agg = torch.cat(H_msgs, dim=-1)
        ## replace code above with single scatter_add for all heads
        N_dst = x_dst.size(0)
        # aggregate all heads at once
        agg = scatter_add(
            out_edges,           # [E, H, d]
            dst_idx,             # [E]
            dim=0,
            dim_size=N_dst
        )                        # [N_dst, H, d]
        # concatenate heads -> [N_dst, H*d]
        agg = agg.reshape(N_dst, -1)


        # project back to in_dim and residual
        out = self.out_lin(self.dropout(agg))
        return out  # [N_dst, in_dim]


# ---------- full hetero GNN ----------
class TEHGT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hid = args.graph_args['embedding_size']
        self.heads = args.graph_args['n_heads']
        self.time_enc_dim = 16 #args['graph_args'].get('time_enc_dim', 16)
        self.n_layers = args.graph_args['n_layers']
        self.dropout = 0.1 #args.graph_args.get('dropout', 0.1)

        # input projectors
        self.lin_text = nn.Linear(args.TEXT_EMBEDDING_SIZES[args.text_embeddings_type], self.hid)
        self.lin_image = nn.Linear(args.IMAGE_EMBEDDING_SIZES[args.image_embeddings_type], self.hid)

        # time encoder
        self.time_encoder = TimeEncoder(self.time_enc_dim)

        # build relation-specific modules
        # For each relation we instantiate an EdgeAwareRelationalAttention
        self.relation_modules = nn.ModuleDict()
        relations = [
            ('text', 'temporal', 'text'),
            ('image', 'temporal', 'image'),
            ('text', 'aligns_with', 'image'),
            ('image', 'aligns_with', 'text'),
            ('text', 'back', 'text'),
            ('image', 'back', 'image'),
            ('text', 'forward', 'text'),
            ('image', 'forward', 'image'),
        ]
        for (s, r, d) in relations:
            name = f"{s}__{r}__{d}"
            self.relation_modules[name] = EdgeAwareRelationalAttention(
                in_dim=self.hid,
                hidden_dim=self.hid,
                heads=self.heads,
                time_emb_dim=self.time_enc_dim,
                dropout=self.dropout,
            )

        # per-layer normalization + FFN
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hid) for _ in range(self.n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hid, self.hid * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hid * 2, self.hid),
                nn.Dropout(self.dropout),
            ) for _ in range(self.n_layers)
        ])

        # final classifier from pooled embedding
        self.classifier = nn.Sequential(
            nn.Linear(self.hid, self.hid // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid // 2, 1)
        )

        ## Fussion modules
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.hid * 2 + 1, self.hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid, 2)
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.hid * 2 + 1, self.hid),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hid, self.hid)
        )

    def forward(self, data):
        device = next(self.parameters()).device

        # initial node embeddings (handle empty node types)
        x_text = data['text'].x.to(device) if 'text' in data.node_types else torch.zeros((0, self.hid), device=device)
        x_image = data['image'].x.to(device) if 'image' in data.node_types else torch.zeros((0, self.hid), device=device)
        # print(x_text.shape, x_image.shape)
        x = {
            'text': self.lin_text(x_text) if x_text.size(0) > 0 else torch.zeros((0, self.hid), device=device),
            'image': self.lin_image(x_image) if x_image.size(0) > 0 else torch.zeros((0, self.hid), device=device)
        }
        # print("numer of labers: ", self.n_layers)
        # print("Edges: ", data.edge_index_dict.keys())
        # print(data.keys())  # ['x', 'ptr', 'edge_index', 'batch', 'edge_attr', 'y']
        # iterate layers
        for layer in range(self.n_layers):
            # collect messages per dst node type
            dst_msgs = {'text': torch.zeros_like(x['text']), 'image': torch.zeros_like(x['image'])}

            # iterate relations and apply relation module
            for key, edge_index in data.edge_index_dict.items():
                src_type, rel, dst_type = key  ## example:  ('text', 'temporal', 'text')
                name = f"{src_type}__{rel}__{dst_type}"
                rel_mod = self.relation_modules[name] if name in self.relation_modules else None #self.relation_modules.get(name, None)
                if rel_mod is None:
                    print(f"Warning: No relation module for {name}, skipping.")
                    continue

                # get nodes & indices
                if src_type not in x or dst_type not in x:
                    print(f"Warning: Missing node type {src_type} or {dst_type}, skipping relation {name}.")
                    continue

                x_src = x[src_type]
                x_dst = x[dst_type]
                # print(name)
                # print(x_src.shape, x_dst.shape)
                eidx = edge_index.to(device)
                # fetch edge_attr if present
                eattr = None
                if hasattr(data, "edge_attrs") and isinstance(data.edge_attrs, dict) and key in data.edge_attrs:
                    # PyG stores edge_attr in data[key].edge_attr; here we access via data[key].edge_attr
                    eattr = data[key].edge_attr.to(device)
                    # encode time
                    eattr = self.time_encoder(eattr)  # [E, time_enc_dim]
                elif hasattr(data[key], 'edge_attr') and data[key].edge_attr is not None:
                    eattr = data[key].edge_attr.to(device)
                    # print(eattr.shape, eattr )
                    eattr = self.time_encoder(eattr)  # assume scalar dt -> encoding
                    # print(eattr.shape)
                else:
                    eattr = None

                if eidx.size(1) == 0:
                    continue
                if x_src.size(0) == 0 or x_dst.size(0) == 0:
                    continue

                msgs = rel_mod(x_src, x_dst, eidx, eattr)  # [N_dst, hid]
                dst_msgs[dst_type] = dst_msgs[dst_type] + msgs

            # finalise node updates with residual + norm + ffn
            for ntype in x.keys():
                if x[ntype].size(0) == 0:
                    continue
                h = x[ntype] + F.dropout(dst_msgs[ntype], p=self.dropout, training=self.training)
                h = self.layer_norms[layer](h)
                # FFN with residual
                h_ffn = self.ffns[layer](h)
                x[ntype] = h + h_ffn

        # Pool into graph-level embedding
        hidden = self.hid
        num_graphs = data.num_graphs
        # text pooling
        if x['text'].size(0) > 0:
            pooled_text = global_max_pool(x['text'], data['text'].batch.to(device), size=num_graphs)  # [G, hid_for_graphs_that_have_text]
        else:
            pooled_text = torch.zeros((num_graphs, hidden), device=device)

        if x['image'].size(0) > 0:
            pooled_image = global_max_pool(x['image'], data['image'].batch.to(device), size=num_graphs)  # [G, hid]
        else:
            pooled_image = torch.zeros((num_graphs, hidden), device=device)
        # handle graphs with no image nodes: replace -inf with 0
        pooled_image = torch.where(torch.isfinite(pooled_image), pooled_image, torch.zeros_like(pooled_image))
   

        # fuse and classify

        # compute per-graph node counts to estimate availability ratio
        ones_text = torch.ones(x['text'].size(0), device=device)
        n_text = scatter_add(ones_text, data['text'].batch.to(device), dim=0, dim_size=num_graphs)
        if x['image'].size(0) > 0:
            ones_img = torch.ones(x['image'].size(0), device=device)
            n_image = scatter_add(ones_img, data['image'].batch.to(device), dim=0, dim_size=num_graphs)
        else:
            n_image = torch.zeros(num_graphs, device=device)
        ratio = torch.where(n_text > 0, n_image / (n_text + 1e-6), torch.zeros_like(n_text))
        ratio = ratio.clamp(0.0, 1.0).unsqueeze(-1)
        # print("Ratio: ", ratio.shape, num_graphs, pooled_text.shape, pooled_image.shape)
        feat = torch.cat([pooled_text, pooled_image, ratio], dim=-1)  # [G, 2H+1]
        logits_fuse = self.fuse_gate(feat)  # [G, 2]
        weights = torch.softmax(logits_fuse, dim=-1)
        w_text = weights[:, 0:1]
        w_image = weights[:, 1:2]
        # combine gated modalities with concatenation-based projection
        weighted_concat = torch.cat([w_text * pooled_text, w_image * pooled_image, ratio], dim=-1)
        h_graph = self.fuse_mlp(weighted_concat)
        

        logits = self.classifier(h_graph)  # [num_graphs, 1]
        probs = torch.sigmoid(logits)
        return {"logits": logits, "probas": probs}

