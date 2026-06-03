import torch
import pprint
import argparse
import nomenclature
from utils import load_args, load_model
torch.multiprocessing.set_sharing_strategy("file_system")
import os
import torch.nn as nn

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

parser = argparse.ArgumentParser(description="Do stuff.")

parser.add_argument("--config_file", type=str, default="./configs/combos/clip_mentalbert_graph.yaml")
parser.add_argument("--model", type=str, default="HGNNv8")
# parser.add_argument("--output_dir", type=str, default="twitter")
parser.add_argument("--dataset", type=str, default="twitter")
parser.add_argument("--window_size", type=int, default=128)
parser.add_argument("--position_embeddings", type=str, default="time2vec")
parser.add_argument("--image_embeddings_type", type=str, default="clip")
parser.add_argument("--text_embeddings_type", type=str, default="mentalbert")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--K", type=int, default=30)
parser.add_argument("--date_threshold", type=int, default=30)


args = parser.parse_args()
args, cfg = load_args(args)


if __name__ == '__main__':
  model = nomenclature.MODELS[args.model](args)
  path = "./checkpoints/{}-{}-ws-{}-{}-{}-K-{}".format(
      args.model,
      args.dataset,
      args.window_size,
      args.image_embeddings_type,
      args.text_embeddings_type,
      args.K,
  )
  model_name = [ fn for fn in os.listdir(path) if fn.startswith("fold-{}".format(args.fold)) ][0]
  checkpoint_path = os.path.join(path, model_name)
  print("::: Loading model from:", checkpoint_path)
  state_dict = torch.load(checkpoint_path, weights_only=True)
  state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
  model.load_state_dict(state_dict)
  model.eval()
  model.train(False)
  model.cuda()
  
  num_params = count_parameters(model)
  print("::: Model parameters:", num_params)
  print("::: Model loaded.")
  evaluator = nomenclature.EVALUATORS["multimodal-evaluator"](args, model)

  results = evaluator.evaluate_graph(save=True)
  print(evaluator.__class__.__name__)
  pprint.pprint(results)
