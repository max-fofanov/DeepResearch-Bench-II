"""Sequential, resumable cascade runner with upstream-compatible score output."""
import argparse
import json
import re
from pathlib import Path

from cascade_judge import CascadeJudge, JudgeError, OpenRouter, exclusive_run, require, save
from gpt_client import get_config
from run_evaluation import PROMPT_TEMPLATE, _extract_docx_content


def collect_reports(report_dir, tasks_path):
    tasks = {}
    with Path(tasks_path).open(encoding='utf-8') as source:
        for line in source:
            if line.strip():
                task = json.loads(line)
                idx = task['idx']
                require(type(idx) is int and idx not in tasks, 'Duplicate/invalid task idx')
                tasks[idx] = task['content']
    require(Path(report_dir).is_dir(), 'Report directory not found')
    reports, seen = [], set()
    for path in sorted(Path(report_dir).glob('*/*')):
        match = re.fullmatch(r'idx-(\d+)\.(md|docx)', path.name, re.IGNORECASE)
        if not match or not path.is_file():
            continue
        idx, model = int(match[1]), path.parent.name
        require((model, idx) not in seen, 'Multiple files for the same model/task')
        require(idx in tasks, f'No task for idx={idx}')
        seen.add((model, idx))
        report = path.read_text(encoding='utf-8') if path.suffix.lower() == '.md' else _extract_docx_content(str(path))
        require(bool(report.strip()), f'Empty/unreadable report: {path}')
        reports.append((model, idx, report, tasks[idx]))
    require(bool(reports), 'No idx-*.md or idx-*.docx reports found')
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description='Jev confidence >=0.8 + DeepSeek with whole-report source review')
    parser.add_argument('--pdf_dir', default='report', help='Per-model Markdown/DOCX directories')
    parser.add_argument('--tasks_jsonl', default='tasks_and_rubrics.jsonl')
    parser.add_argument('--out_jsonl', default='cascade-results.jsonl')
    parser.add_argument('--cache_dir', default='.cascade-cache', help='Persistent raw responses and request journal')
    parser.add_argument('--budget_usd', type=float, default=2, help='Total cumulative cache budget, including retries and unknown bills')
    args = parser.parse_args(argv)
    try:
        reports = collect_reports(args.pdf_dir, args.tasks_jsonl)
        output = Path(args.out_jsonl)
        require(output.resolve() != Path(args.tasks_jsonl).resolve(), 'Output must not overwrite tasks')
        require(output.resolve().suffix == '.jsonl', 'Output must be a JSONL file')
        # A dedicated cache owns its output; do not overwrite another judge's historical results.
        with exclusive_run(args.cache_dir):
            owner_path = Path(args.cache_dir) / 'output.json'
            owner = {'path': str(output.resolve())}
            if owner_path.exists():
                require(json.loads(owner_path.read_text()) == owner, 'Cache output path changed')
            else:
                require(not output.exists(), 'Output exists without this cache; choose a new output')
                save(owner_path, owner)
            judge = CascadeJudge(args.cache_dir, args.budget_usd,
                                 OpenRouter(get_config('OPENROUTER_API_KEY')), PROMPT_TEMPLATE)
            results, failures = [], []
            for model, idx, report, content in reports:
                try:
                    result = judge.evaluate(model, idx, report, content)
                    results.append(dict(model=model, idx=idx, result=result))
                    print(f'[ok] {model} idx={idx}, source_review={result["source_review"]}')
                except JudgeError as exc:
                    failures.append(dict(model=model, idx=idx, error=str(exc)))
                    print(f'[error] {model} idx={idx}: {exc}')
                    # Stop on failure: preserve earlier results and resolve uncertainty before spending more.
                    break
            summary = dict(completed=len(results), total=len(reports), failures=failures, **judge.spending())
            save(Path(args.cache_dir) / 'summary.json', summary)
            # Failed/unvisited reports never become zero scores. Do not replace a previous full export
            # with a partial rerun after inputs were changed or a new failure was encountered.
            if not failures:
                output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.with_name(output.name + '.tmp')
                with temp.open('w', encoding='utf-8') as dest:
                    for result in results:
                        dest.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                temp.replace(output)
            print(json.dumps(summary))
            return 1 if failures else 0
    except (JudgeError, OSError, ValueError, KeyError) as exc:
        # Network exceptions are sanitized inside the transport/request layer.
        print(f'[error] {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
