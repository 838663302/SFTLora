from datasets import load_dataset

import config


def process():
    data_dict = load_dataset(
        "json",
        data_files={
            "train": str(config.DATA_DIR / "train.jsonl"),
            "val": str(config.DATA_DIR / "val.jsonl"),
        },
    )

    return data_dict


if __name__ == "__main__":
    print(process())
