"""Generate sample dataset files, sample C programs, and split JSON files."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from config import DATA_DIR, IR_DIR, MODULES_JSON

C_DIR = DATA_DIR / "c"
C_DIR.mkdir(parents=True, exist_ok=True)
IR_DIR.mkdir(parents=True, exist_ok=True)

# 1. Generate 10 C programs
SAMPLE_PROGRAMS = {
    "sample_loop.c": """
#include <stdio.h>
int main() {
    unsigned int sum = 0;
    for (unsigned int i = 0; i < 1000; i++) {
        sum += i * i;
    }
    printf("%u\\n", sum);
    return 0;
}
""",
    "sample_array.c": """
#include <stdio.h>
int main() {
    unsigned int arr[100];
    for (int i = 0; i < 100; i++) arr[i] = i * 3;
    unsigned int s = 0;
    for (int i = 0; i < 100; i++) s += arr[i];
    printf("%u\\n", s);
    return 0;
}
""",
    "sample_branch.c": """
#include <stdio.h>
int main() {
    unsigned int x = 42, y = 100, z = 0;
    if (x > 20) z = x * 2;
    else z = y + 5;
    if (z % 2 == 0) z += 10;
    printf("%u\\n", z);
    return 0;
}
""",
    "sample_function.c": """
#include <stdio.h>
static unsigned int compute(unsigned int a, unsigned int b) {
    return (a ^ b) + (a * b);
}
int main() {
    unsigned int res = compute(15, 27);
    printf("%u\\n", res);
    return 0;
}
""",
    "sample_math.c": """
#include <stdio.h>
int main() {
    unsigned int a = 1234, b = 5678, c = 9012;
    unsigned int res = (a * b) / (c | 1) + (a ^ b ^ c);
    printf("%u\\n", res);
    return 0;
}
""",
    "sample_nested_loop.c": """
#include <stdio.h>
int main() {
    unsigned int acc = 0;
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < 50; j++) {
            acc += (i + 1) * (j + 1);
        }
    }
    printf("%u\\n", acc);
    return 0;
}
""",
    "sample_switch.c": """
#include <stdio.h>
int main() {
    unsigned int val = 3, res = 0;
    switch (val & 3) {
        case 0: res = 10; break;
        case 1: res = 20; break;
        case 2: res = 30; break;
        default: res = 40; break;
    }
    printf("%u\\n", res);
    return 0;
}
""",
    "sample_dead_code.c": """
#include <stdio.h>
int main() {
    unsigned int x = 10, y = 20;
    unsigned int unused1 = x * 50;
    unsigned int unused2 = y + 100;
    (void)unused1; (void)unused2;
    printf("%u\\n", x + y);
    return 0;
}
""",
    "sample_memory.c": """
#include <stdio.h>
int main() {
    unsigned int a = 5;
    unsigned int *p = &a;
    *p += 15;
    printf("%u\\n", *p);
    return 0;
}
""",
    "sample_mixed.c": """
#include <stdio.h>
static unsigned int helper(unsigned int v) {
    return v * 7 + 3;
}
int main() {
    unsigned int total = 0;
    for (int i = 0; i < 20; i++) {
        if (i % 2 == 0) total += helper(i);
        else total += i * 2;
    }
    printf("%u\\n", total);
    return 0;
}
"""
}

def create_sample_files():
    print("Writing sample C programs to data/c/ ...")
    for name, code in SAMPLE_PROGRAMS.items():
        p = C_DIR / name
        p.write_text(code.strip() + "\n")

    # 2. Write sample_dataset.csv
    csv_path = DATA_DIR / "sample_dataset.csv"
    print(f"Writing {csv_path} ...")
    rows = [
        ["module_id", "program_type", "initial_size", "O2_size", "O3_size", "Oz_size", "best_ours_size", "reduction_percent", "best_pipeline"],
        ["sample_loop", "loop", 120, 65, 60, 58, 45, 62.5, "mem2reg -> loop-rotate -> licm -> dce"],
        ["sample_array", "array", 185, 92, 88, 85, 71, 61.6, "mem2reg -> sroa -> instcombine -> simplifycfg"],
        ["sample_branch", "branch", 95, 54, 52, 50, 41, 56.8, "mem2reg -> simplifycfg -> gvn -> dce"],
        ["sample_function", "function_call", 110, 60, 56, 54, 42, 61.8, "inline -> sroa -> instcombine -> dce"],
        ["sample_math", "arithmetic", 80, 45, 42, 40, 32, 60.0, "reassociate -> instcombine -> simplifycfg"],
        ["sample_nested_loop", "nested_loop", 240, 115, 105, 100, 82, 65.8, "mem2reg -> loop-simplify -> loop-rotate -> licm -> indvars"],
        ["sample_switch", "switch_branch", 105, 58, 55, 52, 43, 59.0, "simplifycfg -> instcombine -> dce"],
        ["sample_dead_code", "dead_code", 130, 40, 38, 36, 28, 78.5, "dce -> adce -> globaldce -> simplifycfg"],
        ["sample_memory", "pointer_ops", 90, 48, 45, 44, 35, 61.1, "sroa -> mem2reg -> instcombine"],
        ["sample_mixed", "mixed_workload", 210, 102, 98, 92, 74, 64.7, "inline -> mem2reg -> sroa -> gvn -> dce -> simplifycfg"]
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # 3. Write train.json, val.json, test.json split files
    if MODULES_JSON.exists():
        mods = json.loads(MODULES_JSON.read_text())
        train_mods, val_mods, test_mods = {}, {}, {}
        for m_id, m_data in mods.items():
            sp = m_data.get("split", "train")
            if sp == "train":
                train_mods[m_id] = m_data
            elif sp == "val":
                val_mods[m_id] = m_data
            else:
                test_mods[m_id] = m_data
        (DATA_DIR / "train.json").write_text(json.dumps(train_mods, indent=2))
        (DATA_DIR / "val.json").write_text(json.dumps(val_mods, indent=2))
        (DATA_DIR / "test.json").write_text(json.dumps(test_mods, indent=2))
        print(f"Splits written: train ({len(train_mods)}), val ({len(val_mods)}), test ({len(test_mods)})")

if __name__ == "__main__":
    create_sample_files()
