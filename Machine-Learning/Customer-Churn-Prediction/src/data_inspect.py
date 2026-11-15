"""Print the dataset audit used by training without modifying the raw CSV."""

import json

from .preprocess import load_dataset


if __name__ == "__main__":
    _, profile = load_dataset()
    print(json.dumps(profile, indent=2))
