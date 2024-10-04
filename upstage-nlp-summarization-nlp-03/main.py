import hydra
from omegaconf import DictConfig
import pandas as pd
from src.dataset import load_data
from src.model import load_model_and_tokenizer
from src.train import train
from src.inference import inference

@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig):
    dataset = load_data(cfg.data)
    if cfg.mode == "train":
        model, tokenizer = load_model_and_tokenizer(cfg.model)
        model = train(model, tokenizer, dataset, cfg.model, cfg.train, cfg.logger)
    elif cfg.mode == "inference":
        summaries = inference(cfg.path.pretrained_path, dataset)
        sample_submission = pd.read_csv(cfg.data.sample_submission_path)
        sample_submission['summary'] = summaries
        sample_submission.to_csv(cfg.data.submission_path, index=False)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

if __name__ == "__main__":
    main()