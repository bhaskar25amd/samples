#!/usr/bin/env python5

import pandas as pd
import json
import argparse
import sys
import os
from datetime import datetime

def parse_from_filename(csv_path):
    base = os.path.basename(csv_path).replace(".csv", "")
    parts = base.split("_")
    #!/usr/bin/env python5

import pandas as pd
import json
import argparse
import sys
import os
from datetime import datetime

def parse_from_filename(csv_path):
    base = os.path.basename(csv_path).replace(".csv", "")
    parts = base.split("_")
    startDelimiter = "inductor",
    endDelimiters = {"torchbench", "huggingface", "timm"}
   
    model_parts = []
    idx = 0

    for p in parts:
        if p.lower() in endDelimiters:
            break
        model_parts.append(p)
        idx += 1

    full_model = "_".join(model_parts[1::])
    remaining = parts[idx:]

    benchmark = remaining[0] if len(remaining) > 0 else "unknown"
    precision = remaining[1] if len(remaining) > 1 else "unknown"
    mode = remaining[2] if len(remaining) > 2 else "unknown"

    return full_model, benchmark, precision, mode


def to_python_int(value):
    """Convert pandas/numpy int64 to native Python int"""
    return int(value)


def process_accuracy_csv(args):
    print("Running pytorch/parse_accuracy_csv.py utility V1.0")

    csv_path = args.csvfile[0]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error reading CSV:", e)
        sys.exit(1)

    if "accuracy" not in df.columns:
        print("Column 'accuracy' not found in CSV.")
        sys.exit(1)

    df["accuracy"] = df["accuracy"].astype(str).str.lower()

    total = to_python_int(len(df))
    skip_count = to_python_int(df["accuracy"].str.contains("skip", na=False).sum())
    pass_count = to_python_int((df["accuracy"] == "pass").sum())
    fail_count = to_python_int(df["accuracy"].str.contains("fail", na=False).sum())

    pass_rate = round((pass_count / total) * 100, 2)
    fail_rate = round((fail_count / total) * 100, 2)
    skip_rate = round((skip_count / total) * 100, 2)

    model, benchmark, precision, mode = parse_from_filename(csv_path)

    k = [
        "schema_version",
        "submit_date",
        "component",
        "subcomp",
        "model",
        "precision",
        "mode",
        "benchmark_infra",
        "version",
        "gpuarch",
        "sdk_version",
        "repo",
        "commithash",
        "total_count",
        "pass_count",
        "version_details",
        "score_details",
        "rocm_details",
        "cuda_details"
    ]

    v = [
        "v1",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Dont Know",
        "accuracy",
        model,
        precision,
        mode,
        benchmark,
        "latest",
        "Dont Know",
        "7.1.0",
        "https://github.com/pytorch/pytorch",
        "8d599045cf4102e451a9e8a9ff215d053ebbe0e8",
        total,
        pass_count,
        {"python": "3.10.12"},
        {
            "total": total,
            "passed": pass_count,
            "nfailures": fail_count,
            "nskipped": skip_count,
            "passrate": pass_rate,
            "failrate": fail_rate,
            "skiprate": skip_rate
        },
        {
            "comment": "<Additional notes>",
            "logurl": "https://github.com/ROCm/aisw-hud/actions/runs/18618539417"
        },
        {
            "comment": "<Additional notes>",
            "logurl": "URL of CUDA run logs"
        }
    ]

    print("total pass fail skip")
    print(total, pass_count, fail_count, skip_count)
    print("passrate =", pass_rate)

    output_json = dict(zip(k, v))
    print("AISWHUD JSON OUTPUT")
    print(json.dumps(output_json, indent=4))

    json_output_file = f"{model}_accuracy.json"
    with open(json_output_file, "w") as f:
        json.dump(output_json, f, indent=4)

    print("JSON written to:", json_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Accuracy CSV File.")
    parser.add_argument('--csvfile', nargs=1, dest='csvfile', required=True,
    help="Path to accuracy CSV file")

    args = parser.parse_args()
    process_accuracy_csv(args)
    sys.exit(0)
