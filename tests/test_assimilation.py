import unittest


class AssimilationTests(unittest.TestCase):
    def test_acceptance_cannot_hide_failed_learning_or_cumulative_forgetting(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('assimilation_run', Path(__file__).resolve().parents[1]/'scripts/assimilation_run.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        config = {'minimum_acquisition': 0.75, 'retention_tolerance': 0.05, 'general_tolerance': 0.05}
        groups = {'A': {'strict_correct': 0.95}, 'B': {'strict_correct': 0.80}}
        self.assertTrue(runner.acceptance(groups, 0.80, 'B', {'A': 1.0}, 0.80, config)[0])
        groups['B']['strict_correct'] = 0.70
        self.assertFalse(runner.acceptance(groups, 0.80, 'B', {'A': 1.0}, 0.80, config)[0])
        groups['B']['strict_correct'] = 0.80
        groups['A']['strict_correct'] = 0.90
        self.assertFalse(runner.acceptance(groups, 0.80, 'B', {'A': 1.0}, 0.80, config)[0])
        groups['A']['strict_correct'] = 0.95
        self.assertFalse(runner.acceptance(groups, 0.70, 'B', {'A': 1.0}, 0.80, config)[0])

    def test_coarsened_kl_zero_for_matching_distribution_and_nonzero_gradient(self):
        import torch
        from continual.assimilation import grouped_kl
        logits = torch.tensor([[1., 2., 3., 4.], [2., 0., -1., 1.]])
        temperature = 2.0
        lp = (logits / temperature).log_softmax(-1)
        values, ids = lp.topk(2, -1)
        teacher = {'indices': ids, 'log_probs': values, 'tail': 1-values.exp().sum(-1)}
        self.assertAlmostEqual(grouped_kl(logits, teacher, temperature).item(), 0., places=5)
        student = torch.zeros_like(logits, requires_grad=True)
        loss = grouped_kl(student, teacher, temperature)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_all_vocabulary_kl_matches_direct_reference(self):
        import torch
        from continual.assimilation import grouped_kl
        t = torch.tensor([[2., -2., 0.]])
        s = torch.tensor([[-1., 1., 0.]], requires_grad=True)
        values, ids = t.log_softmax(-1).topk(3, -1)
        teacher = {'indices': ids, 'log_probs': values, 'tail': torch.zeros(1)}
        expected = torch.nn.functional.kl_div(s.log_softmax(-1), t.softmax(-1), reduction='batchmean')
        self.assertAlmostEqual(grouped_kl(s, teacher, 1).item(), expected.item(), places=5)


if __name__ == '__main__':
    unittest.main()
