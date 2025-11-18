#!/usr/bin/env python3
import pandas as pd
import json
import sys
import os
from datetime import datetime

def parse_from_filename(csv_path):
    """
    Example filename:
    aot_inductor_huggingface_bfloat16_inference_rocm_accuracy.csv
    Extract:
    model       = aot_inductor
    component   = Dont Know
    benchmark   = huggingface
    precision   = bfloat16
    mode        = inference
    gpuarch     = Dont Know
    """
    base = os.path.basename(csv_path).replace(".csv", "")
    parts = base.split("_")

    # adjust depending on your filename convention
    return {
        "model": parts[1],
        "component": "Dont Know",
        "benchmark_infra": parts[3],
        "precision": parts[4],
        "mode": parts[5],
        "gpuarch": "Dont Know"
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_accuracy_json.py <csv_file_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Check accuracy column
    if "accuracy" not in df.columns:
        print("Column 'accuracy' not found in CSV.")
        sys.exit(1)

    # Count accuracy values
    df["accuracy"] = df["accuracy"].astype(str).str.lower()

    # SKIP if "skip" present
    skip_count = df["accuracy"].str.contains("skip", na=False).sum()

    # PASS only if exactly "pass"
    pass_count = (df["accuracy"] == "pass").sum()

    # FAIL if "fail" present
    fail_count = df["accuracy"].str.contains("fail", na=False).sum()

    total = len(df)

    pass_rate = round((pass_count / total) * 100, 2)
    fail_rate = round((fail_count / total) * 100, 2)
    skip_rate = round((skip_count / total) * 100, 2)
    # Extract metadata from filename
    meta = parse_from_filename(csv_path)

    # Build JSON structure
    output = {
        "schema_version": "v1",
        "submit_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "component": meta["component"],
        "subcomp": "accuracy",
        "model": meta["model"],
        "precision": meta["precision"],
        "mode": meta["mode"],
        "benchmark_infra": meta["benchmark_infra"],
        "version": "latest",
        "gpuarch": meta["gpuarch"],
        "sdk_version": "7.1.0",
        "repo": "https://github.com/pytorch/pytorch",
        "commithash": "8d599045cf4102e451a9e8a9ff215d053ebbe0e8",
        "total_count": int(total),
        "pass_count": int(pass_count),
        "score":float(pass_rate),
        "version_details": {
            "python": "3.10.12"
        },
        "score_details": {
            "logurl": "https://github.com/ROCm/aisw-hud/actions/runs/18618539417",
            "total": int(total),
            "passed": int(pass_count),
            "nfailures": int(fail_count),
            "passrate": float(pass_rate),
            "failrate": float(fail_rate),
            "skiprate": float(skip_rate),
            "hostname": "<hostname>",
            "docker": "<docker_name>"
        },
        "rocm_details": {
            "comment": "<Additional notes>",
            "logurl": "https://github.com/ROCm/aisw-hud/actions/runs/18618539417"
        },
        "cuda_details": {
            "comment": "<Additional notes>",
            "logurl": "URL of CUDA run logs"
        }
    }

    # Write JSON output file
    json_output_file = f"{meta['model']}_accuracy.json"
    with open(json_output_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nJSON file generated successfully: {json_output_file}")


if __name__ == "__main__":
    main()
