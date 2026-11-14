"""Evaluate diverse saved lectures without changing shared or personal artifacts.

The inventory covers every saved intervention. The run samples one educational
moment per video, then fills missing classifier-hint types. Hints are not visual
truth. Exit codes indicate technical completion; semantic review stays pending.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from category_normalization import normalize_category_labels


def catalog(data_dir):
    videos, rows, missing = [], [], []
    for source in sorted(data_dir.glob('*/interventions.json')):
        payload = json.loads(source.read_text(encoding='utf-8'))
        records = payload.get('interventions', [])
        videos.append({'video_id': source.parent.name, 'intervention_count': len(records)})
        for record in records:
            frame = Path(record['frame_path'])
            if not frame.is_absolute():
                frame = ROOT / frame
            row = {'video_id': source.parent.name, 'intervention_id': record['id'],
                   'timestamp': record['timestamp'], 'frame_path': str(frame),
                   'classifier_hint': record.get('content_type', 'unknown')}
            row['hint_categories'] = list(normalize_category_labels([row['classifier_hint']]).canonical_categories)
            if not frame.is_file() or not (source.parent / 'transcript.json').is_file():
                missing.append(row)
                continue
            rows.append(row)
    return videos, rows, missing


def select_cases(rows):
    educational = [row for row in rows if row['classifier_hint'] != 'non-educational']
    cases, seen = [], set()
    for video in sorted({row['video_id'] for row in educational}):
        candidates = sorted((row for row in educational if row['video_id'] == video),
                            key=lambda row: (row['timestamp'], row['intervention_id']))
        case = candidates[len(candidates) // 2]
        cases.append(case)
        seen.add((case['video_id'], case['intervention_id']))
    covered = {case['classifier_hint'] for case in cases}
    for hint in sorted({row['classifier_hint'] for row in educational} - covered):
        case = next(row for row in educational if row['classifier_hint'] == hint)
        key = (case['video_id'], case['intervention_id'])
        if key not in seen:
            cases.append(case)
            seen.add(key)
    return cases


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--snapshots', type=Path, nargs='+', help='Isolated extraction reports; later reports override earlier matching cases.')
    parser.add_argument('--model', default='mistral:latest')
    parser.add_argument('--provider', choices=['ollama', 'gemini', 'modal_mistral'], default='ollama')
    parser.add_argument('--modes', nargs='+', default=['brief', 'explanatory', 'detailed'],
                        choices=['brief', 'explanatory', 'detailed'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--case-indices', type=int, nargs='+', help='Reproduce selected numbered cases without overwriting prior runs.')
    args = parser.parse_args()
    if args.provider == 'modal_mistral':
        from modal_inference.mistral_config import MODEL_ID
        args.model = MODEL_ID
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.extract and args.snapshots:
        parser.error('--extract and --snapshots are mutually exclusive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    videos, rows, missing = catalog(ROOT / 'data')
    cases = select_cases(rows)
    if args.case_indices and any(index < 1 or index > len(cases) for index in args.case_indices):
        parser.error('case indices must refer to the numbered inventory cases')
    hints = sorted({category for row in rows for category in row['hint_categories']})
    report = {'videos': videos, 'saved_frame_count': len(rows), 'missing_inputs': missing,
              'classifier_hint_counts': dict(Counter(row['classifier_hint'] for row in rows)),
              'hint_categories': hints,
              'categories_without_unambiguous_hint': sorted(set(f'CT-{i:02d}' for i in range(1, 13)) - set(hints)),
              'selection': 'one middle educational moment per video plus remaining classifier hint types',
              'semantic_review': 'pending; classifier hints are not reviewed ground truth',
              'cases': cases, 'runs': []}

    def save():
        temporary = args.output / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(args.output / 'report.json')

    save()
    print(f'{len(videos)} videos, {len(rows)} available frames, {len(cases)} diverse cases', flush=True)
    if args.inventory_only:
        return 0
    source_runs = {}
    if args.snapshots:
        for source_root in args.snapshots:
            source_report = json.loads((source_root / 'report.json').read_text(encoding='utf-8'))
            for run in source_report['runs']:
                source_case = source_report['cases'][run['case'] - 1]
                source_runs[(source_case['video_id'], source_case['intervention_id'])] = Path(run['folder'])
    failures = 0
    selected = [(index, case) for index, case in enumerate(cases, 1)
                if (args.limit is None or index <= args.limit)
                and (not args.case_indices or index in args.case_indices)]
    report['selected_case_indices'] = [index for index, _ in selected]
    from llm_interface.gemini_provider import DEFAULT_MODEL
    report['text_model'] = (os.environ.get('GEMINI_MODEL') or DEFAULT_MODEL) if args.provider == 'gemini' else args.model
    report['text_provider'] = args.provider
    report['snapshot_reports'] = [str(path) for path in args.snapshots or []]
    save()
    for progress, (index, case) in enumerate(selected, 1):
        folder = args.output / f"{index:02d}-{case['video_id']}-{case['intervention_id'][:8]}"
        command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate_explanation_quality.py'),
                   '--video=' + case['video_id'], '--intervention', case['intervention_id'], '--output', str(folder)]
        if args.extract:
            command.append('--extract')
        if args.extract_only:
            command.append('--extract-only')
        command += ['--modes', *args.modes]
        command += ['--model', args.model]
        command += ['--provider', args.provider]
        if args.snapshots:
            source_folder = source_runs.get((case['video_id'], case['intervention_id']))
            if source_folder is None:
                raise ValueError(f"No isolated extraction for {case['video_id']} {case['intervention_id']}")
            if not source_folder.is_absolute():
                source_folder = ROOT / source_folder
            command += ['--snapshot', str(source_folder / 'snapshot.json')]
        with (args.output / f'{index:02d}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report['runs'].append({'case': index, 'folder': str(folder), 'exit_code': result.returncode,
                               'technical_status': 'completed' if result.returncode == 0 else 'failed',
                               'semantic_review': 'pending'})
        failures += result.returncode != 0
        save()
        print(f"{progress}/{len(selected)} (case {index}): {case['video_id']} {case['classifier_hint']} exit={result.returncode}", flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
