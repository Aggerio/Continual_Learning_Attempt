"""Deterministic synthetic tasks and executable, order-sensitive scoring."""
import hashlib
import json
import random

SPEC = (
    'Execute the JSON record program. Reply only with the resulting JSON array. '
    'Operations run left to right. select: mapping is {old_field:new_field}; '
    'filter: keep rows satisfying field gt or lt value; sort: ascending by field; '
    'group: sum value_field by key_field, emit {key_field:key,"total":sum} '
    'in ascending key order; join: match field against right rows on field, '
    'append their extra fields. Preserve input order except sort and group.'
)


def compact(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def execute(rows, ops):
    rows = [dict(r) for r in rows]
    for op in ops:
        kind = op['op']
        if kind == 'select':
            rows = [{new: r[old] for old, new in op['mapping'].items()} for r in rows]
        elif kind == 'filter':
            if op['cmp'] not in ('gt', 'lt'):
                raise ValueError(op)
            rows = [r for r in rows if (r[op['field']] > op['value'] if op['cmp'] == 'gt'
                                        else r[op['field']] < op['value'])]
        elif kind == 'sort':
            rows = sorted(rows, key=lambda r: r[op['field']])
        elif kind == 'group':
            totals = {}
            for r in rows:
                k = r[op['key_field']]
                totals[k] = totals.get(k, 0) + r[op['value_field']]
            rows = [{op['key_field']: k, 'total': v} for k, v in sorted(totals.items())]
        elif kind == 'join':
            index = {r[op['field']]: r for r in op['right']}
            rows = [{**r, **index[r[op['field']]]} for r in rows]
        else:
            raise ValueError(kind)
    return rows


def make_example(rng, skill, split, serial):
    value_field, key_field = rng.sample(['score', 'cost', 'qty', 'level', 'team', 'zone'], 2)
    values = rng.sample(range(1, 20), rng.randint(2, 4))
    rows = [{key_field: rng.choice(['x', 'y']), value_field: v} for v in values]
    rng.shuffle(rows)
    select = {'op': 'select', 'mapping': {value_field: rng.choice(['amount', 'result', 'value'])}}
    threshold = rng.choice(sorted(values)[:-1])
    filt = {'op': 'filter', 'field': value_field, 'cmp': 'gt', 'value': threshold}
    sort = {'op': 'sort', 'field': value_field}
    group = {'op': 'group', 'key_field': key_field, 'value_field': value_field}
    join = {'op': 'join', 'field': key_field,
            'right': [{key_field: k, 'label': rng.choice(['red', 'blue', 'gold'])} for k in ['x', 'y']]}
    plans = {'A': [select], 'B': [filt], 'C': [sort], 'D': [group], 'E': [join],
             'BA': [filt, select], 'BC': [filt, sort], 'BD': [filt, group], 'BE': [filt, join],
             'BDA': [filt, group, {'op': 'select', 'mapping': {key_field: 'key', 'total': 'sum'}}],
             'BCE': [filt, sort, join]}
    ops = plans[skill]
    program = {'rows': rows, 'ops': ops}
    prompt = compact(program)
    return {'id': f'{split}-{skill}-{serial}', 'skill': skill, 'split': split,
            'prompt': prompt, 'answer': execute(rows, ops),
            'fingerprint': hashlib.sha256(prompt.encode()).hexdigest()}


def generate(config):
    """Global prompt deduplication prevents train/dev/test collisions."""
    output, fingerprints = {}, set()
    for split, count in [('train', config['train_examples']), ('dev', config['dev_examples']),
                         ('test', config['test_examples'])]:
        output[split] = {}
        for skill in ['A', 'B', 'C', 'D', 'E', 'BA', 'BC', 'BD', 'BE', 'BDA', 'BCE']:
            # BDA and BCE never enter learner-accessible training data.
            if split == 'train' and skill in ('BDA', 'BCE'):
                continue
            seed = int(hashlib.sha256(f"{config['seed']}:{split}:{skill}".encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            examples = []
            while len(examples) < count:
                item = make_example(rng, skill, split, len(examples))
                if item['fingerprint'] not in fingerprints:
                    fingerprints.add(item['fingerprint'])
                    examples.append(item)
            output[split][skill] = examples
    return output


def score(text, expected):
    """Strict JSON and an explicitly separate Markdown-tolerant diagnostic."""
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate JSON key')
            result[k] = v
        return result

    def nonfinite(value):
        raise ValueError('Nonfinite JSON value')

    def parse(value):
        return json.loads(value, object_pairs_hook=pairs, parse_constant=nonfinite)

    raw = text.strip()
    try:
        result = parse(raw)
        strict = True
    except ValueError:
        strict = False
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        try:
            result = parse(raw)
        except ValueError:
            return {'strict_correct': False, 'semantic_correct': False, 'valid_json': False}
    # JSON serialization distinguishes bool from int, unlike Python equality.
    correct = compact(result) == compact(expected) if not isinstance(expected, (list, dict)) else (
        json.dumps(result, sort_keys=True) == json.dumps(expected, sort_keys=True))
    return {'strict_correct': strict and correct, 'semantic_correct': correct, 'valid_json': strict}


def general_examples():
    cases = [
        ('What is 7 plus 8? Reply only with a JSON integer.', 15),
        ('What is 9 minus 3? Reply only with a JSON integer.', 6),
        ('What is 4 times 3? Reply only with a JSON integer.', 12),
        ('What is 20 divided by 4? Reply only with a JSON integer.', 5),
        ('Reply only with a JSON array containing the words cat and dog in that order.', ['cat', 'dog']),
        ('Reply only with the JSON boolean for whether 5 is greater than 2.', True),
        ('Reply only with the JSON boolean for whether 3 is greater than 9.', False),
        ('Reply only with a JSON string containing the opposite of hot.', 'cold'),
        ('Reply only with a JSON string containing the lowercase version of HELLO.', 'hello'),
        ('Reply only with a JSON string containing the plural of book.', 'books'),
        ('Reply only with a JSON integer: how many letters are in cat?', 3),
        ('Reply only with a JSON integer: the next number after 2,4,6 is?', 8),
        ('Reply only with a JSON string: which is a fruit, apple or chair?', 'apple'),
        ('Reply only with a JSON string: what language is the word bonjour from?', 'French'),
        ('Reply only with the empty JSON array.', []),
        ('Reply only with a JSON object mapping the key status to the string ready.', {'status': 'ready'}),
    ]
    return [{'id': f'general-{i}', 'skill': 'general', 'prompt': p, 'answer': a,
             'system': 'Follow the user instruction exactly.'} for i, (p, a) in enumerate(cases)]


def general_mc_examples():
    """Fixed multiple-choice probe, independent of JSON response formatting."""
    cases = [
        ('Which animal barks?', ['dog', 'cat', 'fish', 'bird']),
        ('Which is a fruit?', ['apple', 'chair', 'rock', 'spoon']),
        ('What is the opposite of hot?', ['cold', 'warm', 'large', 'bright']),
        ('What is the plural of book?', ['books', 'booked', 'booking', 'bookies']),
        ('Which word means the same as small?', ['tiny', 'huge', 'loud', 'fast']),
        ('Which language does bonjour come from?', ['French', 'English', 'German', 'Spanish']),
        ('Which city is the capital of France?', ['Paris', 'Rome', 'Berlin', 'Madrid']),
        ('Which planet do humans live on?', ['Earth', 'Mars', 'Venus', 'Jupiter']),
        ('How many days are in a week?', ['7', '5', '8', '10']),
        ('What do plants absorb from sunlight?', ['energy', 'sand', 'metal', 'salt']),
        ('Which is normally used to cut paper?', ['scissors', 'pillow', 'cup', 'shoe']),
        ('If all cats are animals, is a cat an animal?', ['yes', 'no', 'never', 'unknown']),
        ('Which number is even?', ['8', '3', '5', '7']),
        ('What comes next: 2, 4, 6, ...?', ['8', '9', '5', '12']),
        ('What is the lowercase form of HELLO?', ['hello', 'HELLO', 'Hello', 'hELLO']),
        ('Which is a programming language?', ['Python', 'Chair', 'Banana', 'Cloudy']),
    ]
    for a, b in [(2, 3), (7, 8), (9, 4), (6, 5), (8, 9), (3, 6), (4, 7), (5, 9)]:
        cases.append((f'What is {a} plus {b}?', [str(a+b), str(a+b+1), str(a+b-1), str(a+b+2)]))
        cases.append((f'What is {a} times {b}?', [str(a*b), str(a*b+1), str(a*b-1), str(a*b+2)]))
    rng = random.Random(73021)
    result = []
    for i, (question, options) in enumerate(cases):
        correct = options[0]
        options = list(options)
        rng.shuffle(options)
        prompt = question + '\n' + '\n'.join(f'{label}. {value}' for label, value in zip('ABCD', options))
        prompt += '\nReply with only the letter of the correct answer.'
        result.append({'id': f'general-mc-{i}', 'question': prompt,
                       'answer': 'ABCD'[options.index(correct)]})
    return result
