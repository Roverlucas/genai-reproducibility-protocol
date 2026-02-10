#!/usr/bin/env python3
"""
Bibliometric analysis of references.bib for the JAIR paper:
"Hidden Non-Determinism in Large Language Model APIs"

Parses the BibTeX file using regex (no external dependencies),
extracts year, entry type, and venue, then computes summary statistics.
"""

import re
import json
from collections import Counter
from pathlib import Path

BIB_PATH = Path('/Users/lucasrover/paper-experiment/article/references.bib')
OUTPUT_PATH = Path('/Users/lucasrover/paper-experiment/analysis/bibliometric_summary.json')


def parse_bib(path: Path) -> list[dict]:
    text = path.read_text(encoding='utf-8')

    entry_pattern = re.compile(
        r'@(\w+)\s*\{([^,]+),\s*(.*?)\n\}',
        re.DOTALL,
    )

    field_pattern = re.compile(
        r'(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
    )

    entries = []
    for m in entry_pattern.finditer(text):
        entry_type = m.group(1).lower()
        cite_key = m.group(2).strip()
        body = m.group(3)

        fields = {}
        for fm in field_pattern.finditer(body):
            field_name = fm.group(1).lower()
            field_value = fm.group(2).strip()
            cleaned = re.sub(r'[{}]', '', field_value)
            cleaned = re.sub(r'\\[a-zA-Z]+\s*', '', cleaned)
            fields[field_name] = cleaned.strip()

        venue = None
        if 'journal' in fields:
            venue = fields['journal']
        elif 'booktitle' in fields:
            venue = fields['booktitle']
        elif 'institution' in fields:
            venue = fields['institution']

        year_str = fields.get('year', None)
        year = int(year_str) if year_str and year_str.isdigit() else None

        entries.append({
            'key': cite_key,
            'type': entry_type,
            'year': year,
            'venue': venue,
            'title': fields.get('title', '(no title)'),
        })

    return entries


def analyze(entries: list[dict]) -> dict:
    total = len(entries)

    years = [e['year'] for e in entries if e['year'] is not None]
    year_counts = dict(sorted(Counter(years).items()))

    type_counts = dict(Counter(e['type'] for e in entries).most_common())

    venues = [e['venue'] for e in entries if e['venue']]
    venue_counts = dict(Counter(venues).most_common(15))

    recent = sum(1 for y in years if 2022 <= y <= 2025)
    recency_pct = round(100 * recent / len(years), 1) if years else 0.0

    min_year = min(years) if years else None
    max_year = max(years) if years else None
    median_year = sorted(years)[len(years) // 2] if years else None

    summary = {
        'total_references': total,
        'year_range': {'min': min_year, 'max': max_year, 'median': median_year},
        'year_distribution': year_counts,
        'entry_type_distribution': type_counts,
        'top_venues': venue_counts,
        'recency_index': {
            'period': '2022-2025',
            'count': recent,
            'total_with_year': len(years),
            'percentage': recency_pct,
        },
    }
    return summary


def print_report(summary: dict) -> None:
    print('=' * 64)
    print('  BIBLIOMETRIC ANALYSIS -- references.bib')
    print('=' * 64)
    print(f"""
  Total references: {summary['total_references']}""")
    yr = summary['year_range']
    print(f"""  Year range:       {yr['min']}--{yr['max']}  (median {yr['median']})""")
    print()

    print('-' * 40)
    print('  Publication Year Distribution')
    print('-' * 40)
    for year, count in sorted(summary['year_distribution'].items()):
        bar = '#' * count
        print(f'    {year}  {count:>3}  {bar}')
    print()

    print('-' * 40)
    print('  Entry Type Distribution')
    print('-' * 40)
    for etype, count in summary['entry_type_distribution'].items():
        print(f'    {etype:<18} {count:>3}')
    print()

    print('-' * 40)
    print('  Top Venues by Frequency')
    print('-' * 40)
    for venue, count in summary['top_venues'].items():
        label = venue if len(venue) <= 55 else venue[:52] + '...'
        print(f'    {count:>2}  {label}')
    print()

    ri = summary['recency_index']
    print('-' * 40)
    print('  Recency Index')
    print('-' * 40)
    print(f'    Period:     {ri["period"]}')
    print(f'    Recent:     {ri["count"]} / {ri["total_with_year"]}')
    print(f'    Percentage: {ri["percentage"]}%')
    print()


def main():
    entries = parse_bib(BIB_PATH)
    summary = analyze(entries)

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + chr(10))
    print(f'[OK] Summary written to {OUTPUT_PATH}')
    print()

    print_report(summary)

    print('-' * 64)
    print('  Full Entry List')
    print('-' * 64)
    for i, e in enumerate(entries, 1):
        venue_str = e['venue'] or '(none)'
        if len(venue_str) > 45:
            venue_str = venue_str[:42] + '...'
        print(f'  {i:>2}. [{e["type"]:<15}] {e["year"] or "????":}  {e["key"]}')
        print(f'       {e["title"][:70]}')
        print(f'       Venue: {venue_str}')
    print()


if __name__ == '__main__':
    main()
