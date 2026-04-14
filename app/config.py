#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch


# Config that serves all environment
GLOBAL_CONFIG = {
    "MODEL_PATH": "../efficint_net_b1_model/model_best_weights.pt",
    "CLASS_MAP_PATH": "../efficint_net_b1_model/class_map.json",
    "IMG_SIZE": (32, 32),
    "NORMALIZE_MEAN": [0.485, 0.456, 0.406],
    "NORMALIZE_STD": [0.229, 0.224, 0.225],
    "SCALE_FACTOR": 255.0,
    "USE_CUDE_IF_AVAILABLE": True,
    "ROUND_DIGIT": 6,
    # RAG pipeline (Mistral API + Tavily)
    "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY", ""),
    "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", ""),
    "LLM_MODEL": os.environ.get("LLM_MODEL", "mistral-small-latest"),
    "MAX_REVISIONS": int(os.environ.get("MAX_REVISIONS", "2")),
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "development": {
        "DEBUG": True
    },

    "staging": {
        "DEBUG": True
    },

    "production": {
        "DEBUG": False,
        "ROUND_DIGIT": 3
    }
}


def get_config() -> dict:
    """
    Get config based on running environment

    :return: dict of config
    """

    # Determine running environment
    ENV = os.environ['PYTHON_ENV'] if 'PYTHON_ENV' in os.environ else 'development'
    ENV = ENV or 'development'

    # raise error if environment is not expected
    if ENV not in ENV_CONFIG:
        raise EnvironmentError(f'Config for envirnoment {ENV} not found')

    config = GLOBAL_CONFIG.copy()
    config.update(ENV_CONFIG[ENV])

    config['ENV'] = ENV
    config['DEVICE'] = 'cuda' if torch.cuda.is_available() and config['USE_CUDE_IF_AVAILABLE'] else 'cpu'

    return config

# load config for import
CONFIG = get_config()

if __name__ == '__main__':
    # for debugging
    import json
    print(json.dumps(CONFIG, indent=4))
