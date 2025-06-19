from pydantic import BaseModel

from ..local.pack import DiskPackedTensors
from .. import types
from .. import dev


class Batch(BaseModel):
    disk_packed_tensors: DiskPackedTensors
    config: types.TrainConfig
    dev_config: dev.TrainConfig
