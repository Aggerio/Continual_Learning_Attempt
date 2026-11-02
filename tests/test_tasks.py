import unittest
from continual.tasks import execute, generate, score


class TaskTests(unittest.TestCase):
    def test_interpreter_composition(self):
        rows = [{'k': 'b', 'v': 4}, {'k': 'a', 'v': 1}, {'k': 'b', 'v': 3}]
        ops = [{'op': 'filter', 'field': 'v', 'cmp': 'gt', 'value': 1},
               {'op': 'group', 'key_field': 'k', 'value_field': 'v'},
               {'op': 'select', 'mapping': {'k': 'key', 'total': 'sum'}}]
        self.assertEqual(execute(rows, ops), [{'key': 'b', 'sum': 7}])
        self.assertEqual(rows[0], {'k': 'b', 'v': 4})

    def test_sort_and_join(self):
        rows = [{'id': 'y', 'n': 8}, {'id': 'x', 'n': 2}]
        ops = [{'op': 'sort', 'field': 'n'}, {'op': 'join', 'field': 'id',
                'right': [{'id': 'x', 'label': 'red'}, {'id': 'y', 'label': 'blue'}]}]
        self.assertEqual(execute(rows, ops), [{'id': 'x', 'n': 2, 'label': 'red'},
                                             {'id': 'y', 'n': 8, 'label': 'blue'}])

    def test_split_isolation_and_reserved_compositions(self):
        cfg = {'seed': 42, 'train_examples': 256, 'dev_examples': 64, 'test_examples': 128}
        data = generate(cfg)
        self.assertEqual(data, generate(cfg))
        fingerprints = [x['fingerprint'] for skills in data.values() for xs in skills.values() for x in xs]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        self.assertNotIn('BDA', data['train'])
        self.assertNotIn('BCE', data['train'])
        import json
        for skills in data.values():
            for xs in skills.values():
                for x in xs:
                    p = json.loads(x['prompt'])
                    self.assertEqual(execute(p['rows'], p['ops']), x['answer'])

    def test_scoring_distinguishes_format_and_semantics(self):
        self.assertTrue(score('[{"x":1}]', [{'x': 1}])['strict_correct'])
        self.assertFalse(score('```json\n[1]\n```', [1])['strict_correct'])
        self.assertTrue(score('```json\n[1]\n```', [1])['semantic_correct'])
        self.assertFalse(score('[2]', [1])['semantic_correct'])
        self.assertFalse(score('[true]', [1])['semantic_correct'])
        self.assertFalse(score('[1,2]', [2,1])['semantic_correct'])
        self.assertFalse(score('answer is 1', 1)['valid_json'])
        self.assertFalse(score('{"x":9,"x":1}', {'x': 1})['strict_correct'])
        self.assertFalse(score('NaN', 1)['valid_json'])

    def test_replay_is_bounded_and_preserves_skill_coverage(self):
        from continual.worker import choose_replay
        class ByteTokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())
        items = [{'id': str(i), 'skill': s, 'answer': i} for s in ['A', 'B'] for i in range(10)]
        config = {'seed': 42, 'replay_examples': 6, 'replay_tokens': 240}
        memory, stats = choose_replay(ByteTokenizer(), items, config)
        self.assertLessEqual(stats['examples'], 6)
        self.assertLessEqual(stats['tokens'], 240)
        self.assertEqual({x['skill'] for x in memory}, {'A', 'B'})
        self.assertEqual(memory, choose_replay(ByteTokenizer(), items, config)[0])


if __name__ == '__main__':
    unittest.main()
