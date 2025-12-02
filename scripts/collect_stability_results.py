#!/usr/bin/env python3
"""
Collect and Summarize Stability Results

Reads all stability.json files from workdir/runs/final/ and creates:
1. CSV file with all results for easy analysis
2. LaTeX table formatted for your report
3. Summary statistics by method combination

Usage:
    python scripts/collect_stability_results.py
    python scripts/collect_stability_results.py --output results.csv
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_config_name(path: str) -> Tuple[str, str, str]:
    """
    Extract dataset, model, and method from path.
    
    Example: 
        workdir/runs/final/wn18rr/rotate/mi_512d/stability.json
        -> ("WN18RR", "RotatE", "MI-512d")
    """
    parts = Path(path).parts
    
    # Extract dataset
    if "wn18rr" in parts:
        dataset = "WN18RR"
    elif "fb15k237" in parts:
        dataset = "FB15k-237"
    else:
        dataset = "Unknown"
    
    # Extract model
    if "rotate" in parts:
        model = "RotatE"
    elif "transe" in parts:
        model = "TransE"
    else:
        model = "Unknown"
    
    # Extract method combination from directory name
    config_dir = parts[-2]  # e.g., "mi_512d" or "med_mi_rscf"
    
    # Parse method
    if config_dir.startswith("med_"):
        # MED combinations
        if "mi_rscf" in config_dir:
            method = "MED+MI+RSCF"
        elif "mi" in config_dir:
            method = "MED+MI"
        elif "rscf" in config_dir:
            method = "MED+RSCF"
        else:
            method = "MED"
    elif config_dir.startswith("mi_rscf_"):
        # MI+RSCF with dimension
        dim = config_dir.split("_")[-1]
        method = f"MI+RSCF-{dim}"
    elif config_dir.startswith("mi_"):
        # MI with dimension
        dim = config_dir.split("_")[-1]
        method = f"MI-{dim}"
    elif config_dir.startswith("rscf_"):
        # RSCF with dimension
        dim = config_dir.split("_")[-1]
        method = f"RSCF-{dim}"
    else:
        method = config_dir
    
    return dataset, model, method


def load_stability_results(base_dir: str = "workdir/runs/final") -> List[Dict]:
    """
    Load all stability.json files from the runs directory.
    
    Returns:
        List of dicts with keys: dataset, model, method, stability, num_pairs
    """
    results = []
    base_path = Path(base_dir)
    
    # Find all stability.json files
    for json_file in base_path.rglob("stability.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Extract stability metrics
            if "stability" in data:
                stability_data = data["stability"]
                stability = stability_data.get("stability", None)
                num_pairs = stability_data.get("num_subgraph_pairs", None)
            else:
                print(f"⚠️  Warning: No stability data in {json_file}")
                continue
            
            # Parse configuration
            dataset, model, method = parse_config_name(str(json_file))
            
            results.append({
                "dataset": dataset,
                "model": model,
                "method": method,
                "stability": stability,
                "num_pairs": num_pairs,
                "path": str(json_file)
            })
            
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    return results


def save_csv(results: List[Dict], output_file: str):
    """Save results to CSV file."""
    import csv
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "model", "method", "stability", "num_pairs", "path"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ CSV saved to: {output_file}")


def generate_latex_table(results: List[Dict], output_file: str):
    """
    Generate LaTeX table for report.
    
    Format:
    | Dataset | Model | MI | RSCF | MI+RSCF | MED+MI | MED+RSCF | MED+MI+RSCF |
    """
    # Group by dataset and model
    table = defaultdict(lambda: defaultdict(dict))
    
    for result in results:
        dataset = result["dataset"]
        model = result["model"]
        method = result["method"]
        stability = result["stability"]
        
        # Extract dimension if present
        if "-" in method:
            method_base, dim = method.rsplit("-", 1)
            key = f"{method_base}_{dim}"
        else:
            key = method
        
        table[dataset][model][key] = stability
    
    # Generate LaTeX
    lines = []
    lines.append("% Stability Results Table")
    lines.append("% Copy this into your report LaTeX file")
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Stability ($C_f$) of KGE Models with Different Method Combinations}")
    lines.append("\\label{tab:stability}")
    lines.append("\\begin{tabular}{ll|ccc|ccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Model} & \\textbf{MI-512d} & \\textbf{RSCF-512d} & \\textbf{MI+RSCF-512d} & \\textbf{MED+MI} & \\textbf{MED+RSCF} & \\textbf{MED+MI+RSCF} \\\\")
    lines.append("\\midrule")
    
    for dataset in ["WN18RR", "FB15k-237"]:
        for i, model in enumerate(["TransE", "RotatE"]):
            if model in table[dataset]:
                data = table[dataset][model]
                
                # Extract values
                mi_512 = data.get("MI_512d", "-")
                rscf_512 = data.get("RSCF_512d", "-")
                mi_rscf_512 = data.get("MI+RSCF_512d", "-")
                med_mi = data.get("MED+MI", "-")
                med_rscf = data.get("MED+RSCF", "-")
                med_mi_rscf = data.get("MED+MI+RSCF", "-")
                
                # Format values
                def fmt(val):
                    if val == "-" or val is None:
                        return "-"
                    return f"{val:.4f}"
                
                # Add row
                if i == 0:
                    lines.append(f"{dataset} & {model} & {fmt(mi_512)} & {fmt(rscf_512)} & {fmt(mi_rscf_512)} & {fmt(med_mi)} & {fmt(med_rscf)} & {fmt(med_mi_rscf)} \\\\")
                else:
                    lines.append(f"& {model} & {fmt(mi_512)} & {fmt(rscf_512)} & {fmt(mi_rscf_512)} & {fmt(med_mi)} & {fmt(med_rscf)} & {fmt(med_mi_rscf)} \\\\")
        
        if dataset == "WN18RR":
            lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Save to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✓ LaTeX table saved to: {output_file}")
    print("\nPreview:")
    print("\n".join(lines[:15]))
    print("...")


def print_summary_statistics(results: List[Dict]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("STABILITY SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    stabilities = [r["stability"] for r in results if r["stability"] is not None]
    
    if not stabilities:
        print("❌ No stability results found!")
        return
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Configurations: {len(results)}")
    print(f"  Mean Stability: {sum(stabilities) / len(stabilities):.4f}")
    print(f"  Min Stability: {min(stabilities):.4f}")
    print(f"  Max Stability: {max(stabilities):.4f}")
    print(f"  Std Dev: {(sum((x - sum(stabilities)/len(stabilities))**2 for x in stabilities) / len(stabilities))**0.5:.4f}")
    
    # Group by method
    method_stats = defaultdict(list)
    for result in results:
        if result["stability"] is not None:
            # Extract base method (remove dimension)
            method = result["method"]
            if "-" in method:
                base_method = method.rsplit("-", 1)[0]
            else:
                base_method = method
            method_stats[base_method].append(result["stability"])
    
    print(f"\n📈 By Method:")
    for method in sorted(method_stats.keys()):
        stabs = method_stats[method]
        avg = sum(stabs) / len(stabs)
        print(f"  {method:20s}: {avg:.4f} (n={len(stabs)})")
    
    # Top 5 most stable configurations
    print(f"\n🏆 Top 5 Most Stable Configurations:")
    sorted_results = sorted(results, key=lambda x: x["stability"] if x["stability"] else 0, reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {result['dataset']:10s} {result['model']:8s} {result['method']:20s}: {result['stability']:.4f}")
    
    # Bottom 5 least stable configurations
    print(f"\n⚠️  Bottom 5 Least Stable Configurations:")
    for i, result in enumerate(sorted_results[-5:], 1):
        print(f"  {i}. {result['dataset']:10s} {result['model']:8s} {result['method']:20s}: {result['stability']:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Collect and summarize stability results")
    parser.add_argument("--base-dir", default="workdir/runs/final", 
                       help="Base directory containing checkpoint folders")
    parser.add_argument("--output-csv", default="stability_results.csv",
                       help="Output CSV file")
    parser.add_argument("--output-latex", default="stability_table.tex",
                       help="Output LaTeX table file")
    args = parser.parse_args()
    
    print("="*60)
    print("Collecting Stability Results")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print("")
    
    # Load results
    results = load_stability_results(args.base_dir)
    
    if not results:
        print("❌ No stability results found!")
        print("\nMake sure you've run: bash scripts/run_stability_evaluation.sh")
        return
    
    print(f"✓ Found {len(results)} stability results\n")
    
    # Save CSV
    save_csv(results, args.output_csv)
    
    # Generate LaTeX table
    generate_latex_table(results, args.output_latex)
    
    # Print summary statistics
    print_summary_statistics(results)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
