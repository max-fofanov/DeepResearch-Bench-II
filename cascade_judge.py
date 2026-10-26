"""Resumable Jev -> DeepSeek rubric judge. No model calls happen at import time."""
import hashlib
import json
import math
import os
from pathlib import Path
from contextlib import contextmanager
from uuid import uuid4

import requests

JEV = 'typesafe/jev-1.13'
DEEPSEEK = 'deepseek/deepseek-v4.1-flash-1017'
PROVIDERS = ['inference-net/fp8', 'nebius/fp8']
CONFIG = dict(version=1, threshold=0.8, jev=JEV, deepseek=DEEPSEEK,
              providers=PROVIDERS, batch_size=50, max_tokens=32768,
              reasoning={'effort': 'medium'}, max_attempts=2,
              data_collection='deny')
SOURCE_CHECK = ('\nSOURCE ATTRIBUTION CHECK: Match blocked references by publication identity as well as title and URL. '
                'A repository mirror, DOI, or alternate URL of the same publication is still the same reference. '
                'A citation introducing a section or table applies to the supported claims within that section or table. '
                'Check every criterion, including claims without an inline citation. '
                'Mere mention of a prohibited URL in a disclaimer does not by itself establish reliance; '
                'use explicit attribution of the supporting content. Apply the original three-way scoring rule '
                'and quote the supporting content and its attribution in evidence. '
                'Do not assume all content in the report is blocked.')
CHOICES = {
    'met': 'Score = 1: The article clearly satisfies the rubric item AND the specific supporting sentence(s) do NOT cite any reference listed in "blocked" (match by title/urls). For numerical data, exact values must be explicitly listed and match the rubric.',
    'not_met': 'Score = 0: The article does NOT mention this rubric item at all.',
    'blocked': 'Score = -1: The article mentions this rubric item, BUT the supporting sentence(s) cite a blocked reference.',
}
SCORES = dict(met=1, not_met=0, blocked=-1)


class JudgeError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise JudgeError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def usage_of(raw):
    usage = raw.get('usage') if isinstance(raw, dict) else None
    return usage if isinstance(usage, dict) else {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as out:
        out.write(encode(value) + '\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


@contextmanager
def exclusive_run(directory):
    """One writer per cache, including separate CLI processes; never steal a lock."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise JudgeError('Cache is locked. Check the recorded PID before removing run.lock.') from None
    try:
        with os.fdopen(fd, 'w') as out:
            out.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def batches(items):
    return [items[i:i + 50] for i in range(0, len(items), 50)]


def rubric_items(content):
    rubric = content.get('rubric')
    require(isinstance(content.get('task'), str) and isinstance(rubric, dict), 'Invalid task/rubric')
    items = []
    for dimension, texts in rubric.items():
        require(dimension in ('info_recall', 'analysis', 'presentation'), 'Unknown rubric dimension')
        require(isinstance(texts, list), 'Rubric dimension must be a list')
        for text in texts:
            require(isinstance(text, str) and bool(text.strip()), 'Empty/non-string criterion')
            items.append(dict(dimension=dimension, text=text))
    require(bool(items), 'Empty rubric')
    require(len({i['text'] for i in items}) == len(items), 'Duplicate criterion text is ambiguous')
    return items


def validate_jev(batch, raw):
    require(raw.get('model') in (JEV, 'typesafe/jev-1.13-20260917') and raw.get('provider') == 'TypeSafe', 'Unexpected Jev model/provider')
    answers = raw.get('answers')
    keys = [f'criterion_{i:03}' for i in range(len(batch))]
    require(isinstance(answers, dict) and set(answers) == set(keys), 'Incorrect Jev coverage')
    rows = []
    for item, key in zip(batch, keys):
        a = answers[key]
        require(isinstance(a, dict) and a.get('type') == 'choice' and a.get('choice') in SCORES, 'Invalid Jev choice')
        p = a.get('probabilities')
        require(isinstance(p, dict) and set(p) == set(SCORES), 'Invalid Jev probabilities')
        require(all(finite(x) and 0 <= x <= 1 for x in p.values()) and abs(sum(p.values()) - 1) <= .02, 'Invalid Jev distribution')
        confidence = a.get('confidence')
        require(confidence is None or (finite(confidence) and 0 <= confidence <= 1), 'Invalid confidence')
        rows.append(dict(item, score=SCORES[a['choice']], confidence=confidence, probabilities=p))
    return rows


def validate_deepseek(batch, raw):
    require(raw.get('model') == DEEPSEEK and raw.get('provider') in ('InferenceNet', 'Nebius'), 'Unexpected DeepSeek model/provider')
    choices = raw.get('choices')
    require(isinstance(choices, list) and len(choices) > 0 and choices[0].get('finish_reason') == 'stop', 'Incomplete DeepSeek response')
    text = choices[0]['message']['content'].strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    rows = json.loads(text)['results']
    require(isinstance(rows, list) and len(rows) == len(batch), 'Incorrect DeepSeek count')
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('rubric_item'), str), 'Invalid criterion')
        require(row['rubric_item'] not in indexed, 'Duplicate result')
        require(type(row.get('score')) is int and row['score'] in (-1, 0, 1), 'Invalid score')
        require(isinstance(row.get('reason'), str) and isinstance(row.get('evidence'), str), 'Missing reason/evidence')
        require(row['score'] == 0 or bool(row['evidence'].strip()), 'Nonzero score without evidence')
        indexed[row['rubric_item']] = row
    require(set(indexed) == {i['text'] for i in batch}, 'Incorrect DeepSeek criterion identity')
    return [dict(indexed[i['text']], dimension=i['dimension']) for i in batch]


class OpenRouter:
    def __init__(self, api_key):
        require(bool(api_key), 'Set OPENROUTER_API_KEY')
        self.api_key = api_key

    def catalog(self, model):
        response = requests.get('https://openrouter.ai/api/v1/models/' + model + '/endpoints', timeout=30)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint, payload):
        # No HTTP retries: a timeout may have been billed by the provider.
        response = requests.post('https://openrouter.ai/api/' + endpoint,
                                 headers={'Authorization': 'Bearer ' + self.api_key,
                                          'Content-Type': 'application/json'},
                                 data=encode(payload).encode(), timeout=(30, 900))
        return response.status_code, response.json()


class CascadeJudge:
    """Caller must hold exclusive_run(cache) for the lifetime of this object."""
    def __init__(self, cache, budget_usd, transport, prompt_template):
        require(finite(budget_usd) and budget_usd > 0, 'Budget must be finite and positive')
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.budget = budget_usd
        self.transport = transport
        self.template = prompt_template
        self.config = dict(CONFIG, prompt_sha256=fingerprint(prompt_template), source_check=SOURCE_CHECK)
        self.catalog_checked = False
        config_path = self.cache / 'judge.json'
        if config_path.exists():
            require(read(config_path) == self.config, 'Judge configuration changed; use a separate cache')
        else:
            save(config_path, self.config)

    def spending(self):
        attempts = [read(p) for p in self.cache.glob('reports/*/*-attempt-*.json')]
        return {'confirmed_usd': sum(a.get('cost_usd', 0) for a in attempts),
                'upper_bound_usd': sum(a.get('cost_usd', a['upper_usd']) for a in attempts),
                'unknown_attempts': sum('cost_usd' not in a for a in attempts),
                'attempts': len(attempts)}

    def check_catalog(self):
        if self.catalog_checked:
            return
        for model, tags, input_price, output_price in [
            (JEV, ['typesafe'], .000000042, 0),
            (DEEPSEEK, PROVIDERS, .0000003, .0000012),
        ]:
            try:
                data = self.transport.catalog(model)
            except Exception:
                raise JudgeError('Provider catalog unavailable; no request dispatched') from None
            for tag in tags:
                route = next((r for r in data['data']['endpoints'] if r['tag'] == tag), None)
                require(route is not None, 'Required provider unavailable: ' + tag)
                ip, op = float(route['pricing']['prompt']), float(route['pricing']['completion'])
                require(finite(ip) and finite(op) and 0 <= ip <= input_price and 0 <= op <= output_price, 'Provider price exceeds budget ceiling: ' + tag)
            save(self.cache / (model.split('/')[0] + '-catalog.json'), data)
        self.catalog_checked = True

    def request(self, folder, stage, index, payload, validator):
        base = folder / f'{stage}-{index}'
        request_file = base.with_suffix('.request.json')
        if request_file.exists():
            require(read(request_file) == payload, 'Cached request changed')
        else:
            save(request_file, payload)
        for attempt in range(CONFIG['max_attempts']):
            path = folder / f'{stage}-{index}-attempt-{attempt}.json'
            if path.exists():
                record = read(path)
                require(record['payload_sha256'] == fingerprint(payload), 'Cached payload changed')
                if record['state'] == 'completed':
                    return validator(record['raw'])
                if record['state'] in ('terminal_error', 'invalid_response'):
                    continue
                raise JudgeError('Unresolved request; inspect before retrying: ' + str(path))
            self.check_catalog()
            size = len(encode(payload).encode()) + 4096
            upper = size * .000000042 if stage == 'jev' else 2 * (size * .0000003 + 32768 * .0000012)
            require(self.spending()['upper_bound_usd'] + upper <= self.budget, 'Budget exhausted (includes unknown billing)')
            record = dict(state='dispatching', payload_sha256=fingerprint(payload), upper_usd=upper)
            save(path, record)
            endpoint = 'alpha/decisions' if stage == 'jev' else 'v1/chat/completions'
            try:
                status, raw = self.transport.post(endpoint, payload)
            except Exception:
                record['state'] = 'outcome_unknown'
                save(path, record)
                raise JudgeError('Network/decoding failure; outcome unknown, not retried: ' + str(path)) from None
            record.update(state='response_saved', http_status=status, raw=raw)
            save(path, record)
            cost = usage_of(raw).get('cost')
            if finite(cost) and cost >= 0:
                record['cost_usd'] = cost
                save(path, record)
                require(cost <= upper, 'Billed cost exceeds reservation; inspect before continuing')
            if status != 200 or (isinstance(raw, dict) and raw.get('error')):
                record['state'] = 'terminal_error'
                if status in (400, 401, 403, 404, 413, 429) and 'cost_usd' not in record:
                    record['cost_usd'] = 0
                if status in (401, 403):
                    record['state'] = 'authorization_error'
                save(path, record)
                require(record['state'] != 'authorization_error', 'OpenRouter authorization failed')
                continue
            require('cost_usd' in record, 'Response cost missing; inspect before continuing')
            try:
                rows = validator(raw)
            except (JudgeError, KeyError, TypeError, ValueError, IndexError, AttributeError):
                record['state'] = 'invalid_response'
                save(path, record)
                continue
            record['state'] = 'completed'
            save(path, record)
            return rows
        raise JudgeError('Two attempts exhausted; inspect saved responses: ' + str(base))

    def evaluate(self, model, idx, report, content):
        items = rubric_items(content)
        require(isinstance(report, str) and bool(report.strip()), 'Empty report')
        folder = self.cache / 'reports' / fingerprint([model, idx])
        manifest = dict(model=model, idx=idx, report_sha256=fingerprint(report),
                        task_sha256=fingerprint(content), config=self.config)
        manifest_path = folder / 'manifest.json'
        if manifest_path.exists():
            require(read(manifest_path) == manifest, 'Report, task or judge changed; use a separate cache')
        else:
            save(manifest_path, manifest)
        # Reconstruct from raw cached responses, also on resume; never trust a stale result alone.
        first = []
        rules = self.template.split('Scoring rule per rubric item (strict):', 1)[1].split('For EACH rubric item', 1)[0]
        rules = 'Scoring rule per rubric item (strict):' + rules.rstrip()
        for i, batch in enumerate(batches(items)):
            state = dict(article=report, task=content['task'], scoring_rules=rules)
            if 'blocked' in content:
                state['blocked'] = content['blocked']
            payload = dict(model=JEV, provider=dict(only=['typesafe'], allow_fallbacks=False, data_collection='deny'),
                           state=state, questions={f'criterion_{k:03}': dict(type='choice',
                           instructions='Assess whether state.article satisfies this rubric item under state.scoring_rules: ' + item['text'],
                           criteria=CHOICES) for k, item in enumerate(batch)})
            first.extend(self.request(folder, 'jev', i, payload, lambda raw, b=batch: validate_jev(b, raw)))
        guard = any(r['score'] == -1 for r in first)
        needed = [r for r in first if guard or r['confidence'] is None or r['confidence'] < .8]
        second = {}
        for i, batch in enumerate(batches(needed)):
            rubric_input = dict(task=content['task'], rubric_items=[r['text'] for r in batch])
            if 'blocked' in content:
                rubric_input['blocked'] = content['blocked']
            prompt = self.template.format(paper=report, rubric=json.dumps(rubric_input, ensure_ascii=False, indent=2))
            if guard:
                prompt += SOURCE_CHECK
            payload = dict(model=DEEPSEEK, messages=[dict(role='user', content=prompt)],
                           reasoning=CONFIG['reasoning'], max_tokens=32768, stream=False,
                           provider=dict(only=PROVIDERS, order=PROVIDERS, allow_fallbacks=True, data_collection='deny'))
            rows = self.request(folder, 'deepseek', i, payload, lambda raw, b=batch: validate_deepseek(b, raw))
            second.update({r['rubric_item']: r for r in rows})
        scores = {dim: {} for dim in ('info_recall', 'analysis', 'presentation')}
        for item in first:
            text = item['text']
            if text in second:
                row = dict(second[text], route='sourceguard' if guard else 'deepseek', jev_score=item['score'], confidence=item['confidence'])
                row.pop('dimension')
                row.pop('rubric_item')
            else:
                row = dict(score=item['score'], route='jev', confidence=item['confidence'], probabilities=item['probabilities'], reason='', evidence='')
            scores[item['dimension']][text] = row
        attempts = [read(p) for p in sorted(folder.glob('*-attempt-*.json'))]
        usage = [usage_of(a.get('raw')) for a in attempts]
        result = dict(task=content['task'], scores=scores, judge=self.config, source_review=guard,
                      usage_summary={'total_tokens': sum(u['total_tokens'] for u in usage if finite(u.get('total_tokens')) and u['total_tokens'] >= 0)},
                      usage_metadata_per_batch=usage,
                      cost_usd=sum(a.get('cost_usd', 0) for a in attempts),
                      cost_upper_bound_usd=sum(a.get('cost_usd', a['upper_usd']) for a in attempts))
        save(folder / 'result.json', result)
        return result
