"""
scripts/preprocess.py
运行: python scripts/preprocess.py
"""
import sys, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataprocess.preprocessor import MOOCCubePreprocessor
from src.dataprocess.splitter import SequenceSplitter


def main():
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir       = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    pre_cfg       = cfg["preprocess"]

    preprocessor = MOOCCubePreprocessor(raw_dir, processed_dir, pre_cfg)
    preprocessor.build_course_db()
    _, user_seqs = preprocessor.build_interactions()
    preprocessor.build_knowledge_graph()
    preprocessor.compute_user_activity(user_seqs)

    splitter = SequenceSplitter(processed_dir)
    splitter.split(user_seqs)
    print("预处理完成。")


if __name__ == "__main__":
    main()
