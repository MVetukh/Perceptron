import json
from dataclasses import dataclass
from enum import Enum, auto


class GradParameter(Enum):
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
            GradParameter.BATCH_SIZE.name.lower(): result['grad_parameters'][GradParameter.BATCH_SIZE.name.lower()],
            GradParameter.EPOCHS.name.lower(): result['grad_parameters'][GradParameter.EPOCHS.name.lower()],
            GradParameter.LEARNING_RATE.name.lower(): result['grad_parameters'][
                GradParameter.LEARNING_RATE.name.lower()]
        }

        # Создаем экземпляр dataclass с загруженными данными
        config = TrainingConfig(**config_data)
    return config


# Пример использования
config = read_param()
print(config.batch_size, config.epochs, config.learning_rate)
