from pydantic import BaseModel

from .. import dev, types
from ..preprocessing.pack import DiskPackedTensors


class Batch(BaseModel):
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    dev_config: dev.TrainConfig
