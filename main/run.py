from inferencePipeline import loadPipeline
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Dict
from inferencePipeline.config import my_path
# Ensure the main directory is in the Python path for imports
_main_dir = Path(__file__).parent
if str(_main_dir) not in sys.path:
    sys.path.insert(0, str(_main_dir))


def main() -> None:
    pipeline: Callable[[List[Dict]], List[Dict]] = loadPipeline()

    # Example driver for local testing only. The evaluation harness will call the
    # returned callable directly and pass the questions list.
    sample_path = os.path.join(
        str(my_path.parent.parent), "data", "sample_questions.jsonl")
    if os.path.exists(sample_path):
        questions: List[Dict] = json.load(open(
            sample_path, "r", encoding="utf-8"))
    else:
        # Minimal stub if no sample is available
        questions = [
            {"questionID": "test-geo-030",
             "subject": "geography",
             "question": "Describe the geography of global internet infrastructure"},
            {"questionID": "test-hist-001",
             "subject": "history",
             "question": "Compare the Roman and Han Empire collapses"},
            {"questionID": "test-alg-001",
             "subject": "algebra",
             "question": "Solve using Gaussian elimination: 3x + 2y = 8, x - y = 1"},
            {"questionID": "test-alg-002",
             "subject": "algebra",
             "question": "Find eigenvalues and eigenvectors of [[2,1],[1,2]]"},
            {"questionID": "test-alg-003",
             "subject": "algebra",
             "question": "Prove the set of all 3×3 diagonal matrices forms a subspace"},
            {"questionID": "test-alg-004",
             "subject": "algebra",
             "question": "Determine if T(x,y,z)=(x+y, y+z, z+x) is invertible"},]
    start_time = time.perf_counter()
    answers = pipeline(questions)
    elapsed_time = time.perf_counter() - start_time

    print(json.dumps({"elapsed_seconds": elapsed_time,
          "num_answers": len(answers)}, ensure_ascii=False))
    # Write answers to a file for quick inspection
    with open("answers.jsonl", "w", encoding="utf-8") as f:
        for ans in answers:
            print(ans)
            f.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
