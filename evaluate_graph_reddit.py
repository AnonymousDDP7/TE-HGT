import torch
import pprint
import argparse
import nomenclature
from utils import load_args, load_model
torch.multiprocessing.set_sharing_strategy("file_system")
import os

parser = argparse.ArgumentParser(description="Do stuff.")

parser.add_argument("--config_file", type=str, default="./configs/combos/clip_mentalbert_graph.yaml")
parser.add_argument("--model", type=str, default="HGNNv8")
# parser.add_argument("--output_dir", type=str, default="reddit")
parser.add_argument("--dataset", type=str, default="reddit")
parser.add_argument("--window_size", type=int, default=128)
parser.add_argument("--position_embeddings", type=str, default="time2vec")
parser.add_argument("--image_embeddings_type", type=str, default="clip")
parser.add_argument("--text_embeddings_type", type=str, default="mentalbert")
# parser.add_argument("--emotion_embeddings_type", type=str, default="j-hartmann")
# parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--date_threshold", type=int, default=10)


args = parser.parse_args()
args, cfg = load_args(args)


if __name__ == '__main__':
  # dataset = nomenclature.DATASETS[args.dataset](  args, kind="test")
  model = nomenclature.MODELS[args.model](args)
  path = "./checkpoints/{}-{}-ws-{}-{}-{}-K-{}-date_threshold-{}".format(
      args.model,
      args.dataset,
      args.window_size,
      args.image_embeddings_type,
      args.text_embeddings_type,
      args.K,
      args.date_threshold,
  )

  model_name = [fn for fn in os.listdir(path) if fn.endswith(".pth")][0]
  checkpoint_path = os.path.join(path, model_name)
  print("::: Loading model from:", checkpoint_path)
  state_dict = torch.load(checkpoint_path, weights_only=True)
  state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
  model.load_state_dict(state_dict)
  model.eval()
  model.train(False)
  model.cuda()
  print("::: Model loaded.")
  evaluator = nomenclature.EVALUATORS["multimodal-evaluator"](args, model)

  results = evaluator.evaluate_graph(save=True)
  print(evaluator.__class__.__name__)
  pprint.pprint(results)
