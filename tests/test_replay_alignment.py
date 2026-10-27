"""Regression tests for transition alignment, including the terminal sentinel row."""

import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer


def record_episode(buffer, first_state, transitions, *, terminated):
    """Mirror recorder row semantics using identifiable scalar states/transitions."""
    env = torch.tensor([0])
    for offset in range(transitions + 1):
        state = first_state + offset
        terminal_row = offset == transitions
        terminal_transition = terminated and offset == transitions - 1
        buffer.add(
            env,
            s=torch.tensor([[float(state)]]),
            a=torch.tensor([[0.0 if terminal_row else float(state + 100)]]),
            a_exec=torch.tensor([[0.0 if terminal_row else float(state + 200)]]),
            xi=torch.tensor([[7.0]]),
            r=torch.tensor([0.0 if terminal_row else float(state + 300)]),
            done=torch.tensor([float(terminated if terminal_row else terminal_transition)]),
            frame=torch.tensor([state + 400]),
            clip=torch.tensor([first_state]),
            terminal=torch.tensor([terminal_row]),
            ep_start=torch.tensor([offset == 0]),
        )
    buffer.end_episode(env)


@pytest.mark.parametrize("terminated", [True, False])
def test_termination_label_belongs_to_executed_transition(terminated):
    buffer = ReplayBuffer(1, 16, 1, 1, 1, "cpu")
    record_episode(buffer, 10, 3, terminated=terminated)

    # Exactly one three-step window, ending at the terminal sentinel state.
    window = buffer.sample_windows(1, 3)
    torch.testing.assert_close(window.states[:, 0, 0], torch.tensor([10., 11., 12., 13.]))
    torch.testing.assert_close(window.done[:, 0], torch.tensor([0., 0., float(terminated)]))
    torch.testing.assert_close(window.reward[:, 0], torch.tensor([310., 311., 312.]))
    torch.testing.assert_close(window.a_clean[:, 0, 0], torch.tensor([110., 111., 112.]))
    torch.testing.assert_close(window.a_exec[:, 0, 0], torch.tensor([210., 211., 212.]))
    torch.testing.assert_close(window.frame[:, 0], torch.tensor([410, 411, 412]))


@pytest.mark.parametrize("capacity", [8, 32])
def test_sampled_windows_preserve_alignment_across_wrap_and_resets(capacity):
    buffer = ReplayBuffer(1, capacity, 1, 1, 1, "cpu")
    record_episode(buffer, 0, 4, terminated=True)
    record_episode(buffer, 20, 5, terminated=True)
    torch.manual_seed(37)
    window = buffer.sample_windows(128, 2)

    source = window.states[:-1, :, 0]
    destination = window.states[1:, :, 0]
    # Windows must never straddle a reset or the overwritten region of the ring.
    torch.testing.assert_close(destination, source + 1)
    expected_done = ((destination == 4) | (destination == 25)).float()
    assert expected_done.any(), "test should exercise a transition into termination"
    assert (expected_done == 0).any(), "test should exercise nonterminal transitions"
    torch.testing.assert_close(window.done, expected_done)
    torch.testing.assert_close(window.reward, source + 300)
    torch.testing.assert_close(window.a_clean[:, :, 0], source + 100)
    torch.testing.assert_close(window.a_exec[:, :, 0], source + 200)
    torch.testing.assert_close(window.frame, (source + 400).long())
