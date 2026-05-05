"""
SequenceSplitter: 按 leave-one-out 切分训练/验证/测试集
"""
import json
from pathlib import Path


class SequenceSplitter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def split(self, user_sequences: dict, min_len: int = 3):
        train, valid, test = {}, {}, {}
        skipped = 0
        for uid, seq in user_sequences.items():
            if len(seq) < min_len:
                skipped += 1
                continue
            train[uid] = seq[:-2]
            valid[uid] = seq[-2:-1]
            test[uid]  = seq[-1:]

        for name, data in [("train", train), ("valid", valid), ("test", test)]:
            path = self.output_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(data, f)

        print(f"数据集切分完成: train={len(train)}, "
              f"valid={len(valid)}, test={len(test)}, skipped={skipped}")
        return train, valid, test
