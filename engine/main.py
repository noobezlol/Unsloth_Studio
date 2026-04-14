import argparse
import json
import sys
from core.config import RunConfig
from backends.unsloth_backend import UnslothBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config_data = json.load(f)
        cfg = RunConfig(**config_data)
    except Exception as e:
        print(json.dumps({"event": "error", "message": str(e)}))
        sys.exit(1)

    try:
        backend = UnslothBackend(cfg)
        backend.load_model()
        backend.train()
    except Exception as e:
        print(json.dumps({"event": "error", "message": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
