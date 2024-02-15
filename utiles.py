import json
from dataclasses import dataclass
from enum import Enum, auto


class LearnParameter(Enum):
    BATCH_SIZE = auto()
    EPOCHS = auto()
    LEARNING_RATE = auto()

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float

def read_param():
    with open("config.json", 'r') as f:
        result = json.load(f)
        config_data = {
            LearnParameter.BATCH_SIZE.name.lower(): result[
                'learning_parameters'][LearnParameter.BATCH_SIZE.name.lower()],
            LearnParameter.EPOCHS.name.lower(): result['learning_parameters'][LearnParameter.EPOCHS.name.lower()],
            LearnParameter.LEARNING_RATE.name.lower(): result['learning_parameters'][
                LearnParameter.LEARNING_RATE.name.lower()]
        }

        # Создаем экземпляр dataclass с загруженными данными
        config:TrainingConfig = TrainingConfig(**config_data)
    return config


if __name__ == "__main__":
    config = read_param()
    print(config.batch_size, config.epochs, config.learning_rate)
