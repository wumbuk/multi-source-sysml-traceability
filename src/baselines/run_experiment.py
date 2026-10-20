"""Run one baseline with experiment_config.json; never overwrite a previous run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from experiment_settings import ROOT, SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=('llm', 'embedding', 'supervised'))
    parser.add_argument('--dataset', default=SETTINGS['dataset'], choices=('Air_Condition', 'Architecting_Spacecraft', 'Extract_Requirement_Values'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='Print commands only; no training or API calls')
    args = parser.parse_args()
    folder = {'llm': 'Llm_baseline', 'embedding': 'Embedding_baseline', 'supervised': 'supervised_baseline'}[args.method]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT / 'results' / folder / args.dataset / f'run_{stamp}_seed{args.seed}'
    gold = ROOT / 'Data' / args.dataset / 'Judge.csv'
    kg = gold.with_name('KG.csv')
    env = os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8', EXPERIMENT_INPUT=str(gold), EXPERIMENT_OUTPUT_DIR=str(output), LLM_PREDICTIONS_FILENAME='llm_predictions.json')
    py = sys.executable
    if args.method == 'llm':
        commands = [[py, '-u', str(ROOT / 'Llm_baseline/main.py')],
                    [py, str(ROOT / 'Llm_baseline/parse_results.py'), '--gold_csv', str(gold), '--pred_json', str(output / 'llm_predictions.json'), '--output_dir', str(output), '--output_metrics', str(output / 'metrics.json'), '--output_mismatch', str(output / 'mismatch.json')]]
    else:
        if args.method == 'embedding':
            code = ROOT / 'Embedding_matched_experiment'
            prediction = output / 'match_results.csv'
            command = [py, '-u', str(code / 'main.py'), '--kg-samples', str(kg), '--query-pairs', str(gold), '--output', str(prediction), '--seed', str(args.seed)]
            evaluation = [py, str(code / 'analyze_match_results.py'), '--match_csv', str(prediction)]
        else:
            code = ROOT / 'Supervised_experiment'
            prediction = output / 'judge_predictions.csv'
            command = [py, '-u', str(code / 'quanliang_main.py'), '--train-path', str(kg), '--test-path', str(gold), '--output-dir', str(output), '--seed', str(args.seed)]
            evaluation = [py, str(code / 'analyze_predictions.py'), '--predictions_csv', str(prediction)]
        evaluation += ['--mode', 'both', '--rq2_setting', 'pipeline', '--pipeline_eval', 'conditional', '--output_metrics', str(output / 'metrics.json'), '--output_mismatch', str(output / 'mismatch.json')]
        commands = [command, evaluation]
    manifest = {'method': args.method, 'dataset': args.dataset, 'seed': args.seed,
                'config': SETTINGS, 'commands': commands, 'output_dir': str(output),
                'note': 'LLM seed is a run label only; no sampling seed is sent. Environment overrides may apply.'}
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if not gold.is_file() or (args.method != 'llm' and not kg.is_file()):
        raise FileNotFoundError('Required dataset input is missing')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run_config.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Output: {output}', flush=True)
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__':
    main()
