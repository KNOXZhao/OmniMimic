"""CPU-only regression coverage for early-failure benchmark normalization."""

import unittest

from supermimic.eval.metrics import return_scores


class BenchmarkMetricsTest(unittest.TestCase):
    def test_early_failure_cannot_score_like_a_completed_clip(self):
        # A policy tracks perfectly for 20 of 100 transitions, then fails. The
        # conditional metric alone reported 1.0 despite losing 80% of the motion.
        early = return_scores(reward_sum=2.0, frames=101, executed_steps=20)
        complete = return_scores(reward_sum=10.0, frames=101, executed_steps=100)
        self.assertAlmostEqual(early.executed_steps_norm_return, 1.0)
        self.assertAlmostEqual(early.full_clip_norm_return, 0.2)
        self.assertAlmostEqual(early.completion_fraction, 0.2)
        self.assertAlmostEqual(complete.full_clip_norm_return, 1.0)
        self.assertAlmostEqual(complete.executed_steps_norm_return, 1.0)
        self.assertLess(early.full_clip_norm_return, complete.full_clip_norm_return)


    def test_penalties_are_preserved_and_no_steps_score_zero(self):
        negative = return_scores(reward_sum=-0.5, frames=11, executed_steps=5)
        self.assertAlmostEqual(negative.full_clip_norm_return, -0.5)
        self.assertAlmostEqual(negative.executed_steps_norm_return, -1.0)
        empty = return_scores(reward_sum=0.0, frames=11, executed_steps=0)
        self.assertEqual(empty.full_clip_norm_return, 0.0)
        self.assertEqual(empty.executed_steps_norm_return, 0.0)


    def test_extra_terminal_hold_step_is_rejected_from_score_horizon(self):
        # A 101-frame clip has exactly 100 transitions; a framework's extra reset
        # bookkeeping step must never inflate the accumulated benchmark reward.
        with self.assertRaisesRegex(ValueError, "transition horizon"):
            return_scores(reward_sum=10.1, frames=101, executed_steps=101)


    def test_invalid_clip_horizons_fail_fast(self):
        for frames, steps in [(1, 0), (10, -1), (10, 10)]:
            with self.subTest(frames=frames, steps=steps), self.assertRaises(ValueError):
                return_scores(0.0, frames, steps)


if __name__ == "__main__":
    unittest.main()
