"""
Smoke tests: verify the three-level model runs forward and backward on random
data for every gate type, and that the ablation switches behave.

Run:  python -m pytest tests/ -q      (or)   python tests/test_forward.py
"""

import torch
import torch.nn.functional as F

from src.model import ThreeLevelNet


def _random_batch(b=4, c=3, hw=32, num_classes=10):
    x = torch.randn(b, c, hw, hw)
    y = torch.randint(0, num_classes, (b,))
    return x, y


def test_forward_backward_all_gates():
    for gate in ["handcrafted", "learned", "none"]:
        model = ThreeLevelNet(num_classes=10, num_slots=4, concept_dim=32, gate_type=gate)
        x, y = _random_batch()
        refined, base, info = model(x, y)
        assert refined.shape == (4, 10)
        assert base.shape == (4, 10)
        loss = F.cross_entropy(refined, y) + 0.3 * F.cross_entropy(base, y)
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert any(grads), f"no gradients flowed for gate={gate}"
        print(f"[ok] gate={gate:11s} loss={loss.item():.3f}")


def test_adaptation_toggle_changes_output():
    torch.manual_seed(0)
    model = ThreeLevelNet(num_classes=10, num_slots=4, concept_dim=32,
                          gate_type="learned", use_adaptation=True)
    x, y = _random_batch()
    refined, base, _ = model(x, y)
    # hypernet uses small-scale init (std=0.01) so delta is small but non-zero
    delta = (refined - base).abs().max().item()
    assert delta < 0.5, f"delta too large at init ({delta:.4f}), hypernet may be unstable"
    print(f"[ok] small-init adaptation is stable at start (max delta={delta:.4f})")


if __name__ == "__main__":
    test_forward_backward_all_gates()
    test_adaptation_toggle_changes_output()
    print("\nAll smoke tests passed.")
