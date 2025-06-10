from art.local.checkpoints import get_step
from art.model import TrainableModel
from art.utils.output_dirs import get_model_dir


def get_model_step(model: TrainableModel, art_path: str) -> int:
    return get_step(get_model_dir(model=model, art_path=art_path))
