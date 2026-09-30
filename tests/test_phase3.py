"""Focused tests for the new objectives and the lightweight architecture."""

import unittest

import torch

from src.losses import pearson_loss, ranking_loss, training_loss
from src.model import build_model


class Phase3Tests(unittest.TestCase):
    def test_ranking_direction_and_epsilon(self):
        labels = torch.tensor([0.1, 0.5, 0.9])
        ordered = torch.tensor([0.2, 0.4, 0.8])
        reversed_scores = torch.tensor([0.8, 0.4, 0.2], requires_grad=True)
        self.assertEqual(ranking_loss(ordered, labels, 0.01).item(), 0)
        loss = ranking_loss(reversed_scores, labels, 0.01)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(reversed_scores.grad).all())
        self.assertEqual(ranking_loss(reversed_scores, labels, 2.0).item(), 0)

    def test_pearson_direction_and_stability(self):
        labels = torch.tensor([0.1, 0.5, 0.9])
        self.assertLess(pearson_loss(labels, labels).item(), 1e-6)
        self.assertGreater(pearson_loss(-labels, labels).item(), 1.9)
        constant = torch.ones(3, requires_grad=True)
        loss = pearson_loss(constant, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(constant.grad).all())

    def test_single_factor_objectives(self):
        pred = torch.tensor([0.3, 0.1, 0.6], requires_grad=True)
        label = torch.tensor([0.2, 0.4, 0.8])
        base = training_loss(pred, label, {})
        self.assertGreater(training_loss(pred, label, {"loss_type": "ranking",
            "ranking_weight": 0.25, "epsilon_rank": 0.01}).item(), base.item())
        self.assertGreater(training_loss(pred, label, {"loss_type": "pearson",
            "pearson_weight": 0.5}).item(), base.item())

    def test_multiscale_forward_and_parameters(self):
        model = build_model(multi_scale=True).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(output.shape, (1, 1))
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 5_000_000)


if __name__ == "__main__":
    unittest.main()
