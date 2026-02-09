#!/usr/bin/env python3
"""
Generate visualization plots for JSON extraction quality metrics.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(json_path):
    """Load metrics from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_validity_and_compliance(metrics, output_path):
    """Plot validity and compliance rates across models and conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Organize data by model
    llama_data = [m for m in metrics if m['model'] == 'llama3']
    gpt_data = [m for m in metrics if m['model'] == 'gpt4']

    # Sort by condition
    llama_data.sort(key=lambda x: x['condition'])
    gpt_data.sort(key=lambda x: x['condition'])

    # Extract metrics
    llama_conditions = [m['condition'] for m in llama_data]
    llama_raw = [m['overall_validity']['json_validity_rate_raw'] * 100 for m in llama_data]
    llama_extracted = [m['overall_validity']['json_validity_rate_extracted'] * 100 for m in llama_data]
    llama_compliant = [m['overall_compliance']['schema_compliance_rate'] * 100 for m in llama_data]

    gpt_conditions = [m['condition'] for m in gpt_data]
    gpt_raw = [m['overall_validity']['json_validity_rate_raw'] * 100 for m in gpt_data]
    gpt_extracted = [m['overall_validity']['json_validity_rate_extracted'] * 100 for m in gpt_data]
    gpt_compliant = [m['overall_compliance']['schema_compliance_rate'] * 100 for m in gpt_data]

    # Plot LLaMA 3
    x_llama = np.arange(len(llama_conditions))
    width = 0.25

    ax1.bar(x_llama - width, llama_raw, width, label='Raw Valid', color='#3498db', alpha=0.8)
    ax1.bar(x_llama, llama_extracted, width, label='Extracted Valid', color='#2ecc71', alpha=0.8)
    ax1.bar(x_llama + width, llama_compliant, width, label='Schema Compliant', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('LLaMA 3 8B: JSON Validity & Compliance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_llama)
    ax1.set_xticklabels(llama_conditions, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 105])
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Plot GPT-4
    x_gpt = np.arange(len(gpt_conditions))

    ax2.bar(x_gpt - width, gpt_raw, width, label='Raw Valid', color='#3498db', alpha=0.8)
    ax2.bar(x_gpt, gpt_extracted, width, label='Extracted Valid', color='#2ecc71', alpha=0.8)
    ax2.bar(x_gpt + width, gpt_compliant, width, label='Schema Compliant', color='#e74c3c', alpha=0.8)

    ax2.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('GPT-4: JSON Validity & Compliance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_gpt)
    ax2.set_xticklabels(gpt_conditions, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_field_emr(metrics, output_path):
    """Plot field-level exact match rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Organize data
    llama_data = [m for m in metrics if m['model'] == 'llama3']
    gpt_data = [m for m in metrics if m['model'] == 'gpt4']

    llama_data.sort(key=lambda x: x['condition'])
    gpt_data.sort(key=lambda x: x['condition'])

    fields = ['objective', 'method', 'key_result', 'model_or_system', 'benchmark']
    field_labels = ['Objective', 'Method', 'Key Result', 'Model/System', 'Benchmark']

    # LLaMA 3 data
    llama_conditions = [m['condition'] for m in llama_data]
    llama_field_data = {f: [] for f in fields}

    for m in llama_data:
        for f in fields:
            val = m['field_accuracy']['field_accuracy'].get(f)
            llama_field_data[f].append(val if val is not None else 0)

    # GPT-4 data
    gpt_conditions = [m['condition'] for m in gpt_data]
    gpt_field_data = {f: [] for f in fields}

    for m in gpt_data:
        for f in fields:
            val = m['field_accuracy']['field_accuracy'].get(f)
            gpt_field_data[f].append(val if val is not None else 0)

    # Plot LLaMA 3
    x_llama = np.arange(len(llama_conditions))
    width = 0.15

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (field, label) in enumerate(zip(fields, field_labels)):
        offset = width * (i - 2)
        ax1.bar(x_llama + offset, llama_field_data[field], width, label=label, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Exact Match Rate', fontsize=11, fontweight='bold')
    ax1.set_title('LLaMA 3 8B: Field-Level EMR', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_llama)
    ax1.set_xticklabels(llama_conditions, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 0.35])
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    # Plot GPT-4
    x_gpt = np.arange(len(gpt_conditions))

    for i, (field, label) in enumerate(zip(fields, field_labels)):
        offset = width * (i - 2)
        ax2.bar(x_gpt + offset, gpt_field_data[field], width, label=label, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Exact Match Rate', fontsize=11, fontweight='bold')
    ax2.set_title('GPT-4: Field-Level EMR', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_gpt)
    ax2.set_xticklabels(gpt_conditions, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 0.35])
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_completeness_rates(metrics, output_path):
    """Plot field completeness (non-empty) rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Organize data
    llama_data = [m for m in metrics if m['model'] == 'llama3']
    gpt_data = [m for m in metrics if m['model'] == 'gpt4']

    llama_data.sort(key=lambda x: x['condition'])
    gpt_data.sort(key=lambda x: x['condition'])

    fields = ['objective', 'method', 'key_result', 'model_or_system', 'benchmark']
    field_labels = ['Objective', 'Method', 'Key Result', 'Model/System', 'Benchmark']

    # LLaMA 3 data
    llama_conditions = [m['condition'] for m in llama_data]
    llama_field_data = {f: [] for f in fields}

    for m in llama_data:
        for f in fields:
            val = m['field_presence']['per_field_non_empty_rate'].get(f, 0)
            llama_field_data[f].append(val * 100)

    # GPT-4 data
    gpt_conditions = [m['condition'] for m in gpt_data]
    gpt_field_data = {f: [] for f in fields}

    for m in gpt_data:
        for f in fields:
            val = m['field_presence']['per_field_non_empty_rate'].get(f, 0)
            gpt_field_data[f].append(val * 100)

    # Plot LLaMA 3
    x_llama = np.arange(len(llama_conditions))
    width = 0.15

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, (field, label) in enumerate(zip(fields, field_labels)):
        offset = width * (i - 2)
        ax1.bar(x_llama + offset, llama_field_data[field], width, label=label, color=colors[i], alpha=0.8)

    ax1.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Non-Empty Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('LLaMA 3 8B: Field Completeness', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_llama)
    ax1.set_xticklabels(llama_conditions, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 105])
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(axis='y', alpha=0.3)

    # Plot GPT-4
    x_gpt = np.arange(len(gpt_conditions))

    for i, (field, label) in enumerate(zip(fields, field_labels)):
        offset = width * (i - 2)
        ax2.bar(x_gpt + offset, gpt_field_data[field], width, label=label, color=colors[i], alpha=0.8)

    ax2.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Non-Empty Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('GPT-4: Field Completeness', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_gpt)
    ax2.set_xticklabels(gpt_conditions, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_overall_emr_comparison(metrics, output_path):
    """Plot overall EMR comparison between models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Organize data
    llama_data = [m for m in metrics if m['model'] == 'llama3']
    gpt_data = [m for m in metrics if m['model'] == 'gpt4']

    llama_data.sort(key=lambda x: x['condition'])
    gpt_data.sort(key=lambda x: x['condition'])

    # Extract overall EMR
    llama_conditions = [m['condition'] for m in llama_data]
    llama_emr = [m['field_accuracy']['overall_field_emr'] * 100 for m in llama_data]

    gpt_conditions = [m['condition'] for m in gpt_data]
    gpt_emr = [m['field_accuracy']['overall_field_emr'] * 100 for m in gpt_data]

    # Combine conditions (union of both)
    all_conditions = sorted(set(llama_conditions + gpt_conditions))

    # Align data
    llama_aligned = []
    gpt_aligned = []

    for cond in all_conditions:
        if cond in llama_conditions:
            idx = llama_conditions.index(cond)
            llama_aligned.append(llama_emr[idx])
        else:
            llama_aligned.append(0)

        if cond in gpt_conditions:
            idx = gpt_conditions.index(cond)
            gpt_aligned.append(gpt_emr[idx])
        else:
            gpt_aligned.append(0)

    # Plot
    x = np.arange(len(all_conditions))
    width = 0.35

    ax.bar(x - width/2, llama_aligned, width, label='LLaMA 3 8B', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, gpt_aligned, width, label='GPT-4', color='#3498db', alpha=0.8)

    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overall Field EMR (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Field-Level Exact Match Rate: LLaMA 3 vs GPT-4', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_conditions, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (llama_val, gpt_val) in enumerate(zip(llama_aligned, gpt_aligned)):
        if llama_val > 0:
            ax.text(i - width/2, llama_val + 0.2, f'{llama_val:.1f}', ha='center', va='bottom', fontsize=8)
        if gpt_val > 0:
            ax.text(i + width/2, gpt_val + 0.2, f'{gpt_val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution."""
    project_root = Path(__file__).parent.parent
    metrics_file = project_root / "analysis" / "json_extraction_metrics.json"
    output_dir = project_root / "analysis" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics...")
    metrics = load_metrics(metrics_file)

    print("\nGenerating plots...")

    plot_validity_and_compliance(
        metrics,
        output_dir / "json_validity_compliance.png"
    )

    plot_field_emr(
        metrics,
        output_dir / "field_level_emr.png"
    )

    plot_completeness_rates(
        metrics,
        output_dir / "field_completeness.png"
    )

    plot_overall_emr_comparison(
        metrics,
        output_dir / "overall_emr_comparison.png"
    )

    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"  - json_validity_compliance.png")
    print(f"  - field_level_emr.png")
    print(f"  - field_completeness.png")
    print(f"  - overall_emr_comparison.png")


if __name__ == "__main__":
    main()
