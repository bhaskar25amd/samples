#!/usr/bin/env python3

import pandas as pd
import json
import argparse
import sys
import os
import re
import shlex
import subprocess
import pathlib
from datetime import datetime

def get_gfxarch():
    gfx_to_prodname = {"gfx90a" : "MI250", "gfx942" : "MI300X", "gfx941" : "MI300A", "gfx950" : "MI350"}
    # pipe amdgpu-arch to uniq: amdgpu-arch | uniq
    archcmd = "amdgpu-arch"
    uniqcmd = "uniq"
    try:
        ps1 = subprocess.Popen(shlex.split(archcmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ps2 = subprocess.Popen(shlex.split(uniqcmd), stdin=ps1.stdout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ps1.stdout.close()
        out = ps2.communicate()[0]
        #out = str(ps1.decode('utf-8')).strip()
        #print(out.decode('utf-8'))
        gfx = str(out.decode('utf-8').strip())
        if gfx in gfx_to_prodname.keys():
            return gfx_to_prodname[gfx]
        return "gfx900"
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
        return "gfx900"
def get_uname():
    uname_result = os.uname()
    uname_str = ' '.join(uname_result)
    return uname_str

def get_commithash():
    cmd = "git rev-parse HEAD"
    try:
        ps1 = subprocess.check_output(shlex.split(cmd), bufsize=0, stderr=subprocess.STDOUT)
        out = str(ps1.decode('utf-8')).strip()
        return str(out)
    except subprocess.CalledProcessError as err:
        for line in str.splitlines(err.output.decode('utf-8')):
            print(line)
        return "error_not_available"


def parse_from_filename(csv_path):
    base = os.path.basename(csv_path).replace(".csv", "")
    parts = base.split("_")
    startDelimiter={"inductor"}
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

    remaining = parts[idx:]   # [benchmark, models/precision, ...]

    # Default
    benchmark = "unknown"
    precision = "unknown"
    mode = "unknown"

    if len(remaining) >= 1:
        benchmark = remaining[0]

    if len(remaining) >= 2:
        # CASE: benchmark, models, precision, mode
        if remaining[1] == "models":
            if len(remaining) >= 3:
                precision = remaining[2]
            if len(remaining) >= 4:
                mode = remaining[3]

        # CASE: benchmark, precision, mode
        else:
            precision = remaining[1]
            if len(remaining) >= 3:
                mode = remaining[2]

    return full_model, benchmark, precision, mode


def to_python_int(value):
    return int(value)


def generate_accuracy_json_from_csv(csv_path, args):
    print(f"\nProcessing CSV: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print("Error reading CSV:", e)
        return None

    if "accuracy" not in df.columns:
        print("Skipping", csv_path, "(no accuracy column)")
        return None

    df["accuracy"] = df["accuracy"].astype(str).str.lower()

    total = to_python_int(len(df))
    skip_count = to_python_int(df["accuracy"].str.contains("skip", na=False).sum())
    pass_count = to_python_int((df["accuracy"] == "pass").sum())
    fail_count = to_python_int(df["accuracy"].str.contains("fail", na=False).sum())

    pass_rate = round((pass_count / total) * 100, 2)
    fail_rate = round((fail_count / total) * 100, 2)
    skip_rate = round((skip_count / total) * 100, 2)

    model, benchmark, precision, mode = parse_from_filename(os.path.basename(csv_path))
    gpuarch = get_gfxarch()
    uname_details = get_uname()
    commithash = get_commithash()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    compname = "pytorch"

    # JSON keys
    k = [
        "schema_version", "submit_date", "component", "subcomp", "model",
        "precision", "mode", "benchmark_infra", "version", "gpuarch",
        "sdk_version", "repo", "commithash", "total_count", "pass_count",
        "version_details", "score_details", "rocm_details", "cuda_details"
    ]

    # JSON values
    v = [
        "v1",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        compname,
        args.subcomp[0],
        model,
        precision,
        mode,
        benchmark,
        args.version[0],
        #args.gpuarch[0]
        gpuarch,
        "7.1.0",
        args.repo[0],
        commithash,
        total,
        pass_count,
        {"python": python_version},
        {
            "total": total,
            "passed": pass_count,
            "nfailures": fail_count,
            "nskipped": skip_count,
            "passrate": pass_rate,
            "failrate": fail_rate,
            "skiprate": skip_rate,
            "hostname": uname_details,
            "docker": args.docker[0]
        },
        {
            "comment": "<Additional notes>",
            "logurl": "https://github.com/ROCm/aisw-hud/actions/runs/" + args.run_id[0]
        },
        {
            "comment": "<Additional notes>",
            "logurl": "URL of CUDA run logs"
        }
    ]

    print("AISWHUD JSON OUTPUT")
    print(json.dumps(dict(zip(k,v)), indent=4))

def process_accuracy_csv(args):
    print("Running pytorch/parse_accuracy_csv.py utility V1.0")
    # --csvdir provided (directory)
    if args.csvdir:
        csv_path = pathlib.Path(args.csvdir[0])
        if not csv_path.exists():
            print("Directory does not exist:", csv_path)
            sys.exit(1)

        csv_files = list(csv_path.glob("*.csv"))
        if not csv_files:
            print("No CSV files found in directory.")
            sys.exit(0)

        for csv_file in csv_files:
            generate_accuracy_json_from_csv(str(csv_file), args)
        return

    print("ERROR: Provide either --csvfile or --csvdir")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Accuracy CSV Files.")
    parser.add_argument('--subcomp', nargs=1, required=True, help="Subcomponent name")
    #parser.add_argument('--gpuarch', nargs=1, required=True, help="gpuarch name")
    parser.add_argument('--run_id', nargs=1, required=True, help="workflow run_id")
    parser.add_argument('--repo', nargs=1, required=True, help="github repo path")
    parser.add_argument('--docker', nargs=1, dest='docker', required=True, help="docker image name")
    parser.add_argument('--version', nargs=1, dest='version', required=True, help="Code version/branch name")
    parser.add_argument('--csvdir', nargs=1, help="Directory containing multiple CSV files")

    args = parser.parse_args()
    process_accuracy_csv(args)
    sys.exit(0)

