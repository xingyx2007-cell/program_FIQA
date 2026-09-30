"""Measure the plain baseline at the competition's 224x224 input size."""

import argparse
import json
from math import prod
from pathlib import Path

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
import torch

from src.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Profile the architecture loaded from this checkpoint")
    parser.add_argument("--multi-scale", action="store_true")
    args = parser.parse_args()
    model = build_model(use_pretrained=False, multi_scale=args.multi_scale).eval()
    checkpoint_epoch = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        checkpoint_epoch = checkpoint["epoch"]
    sample = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        output = model(sample)
    analysis = FlopCountAnalysis(model, sample)
    analysis.unsupported_ops_warnings(False)
    flops = int(analysis.total())
    unsupported = {str(name): int(count) for name, count in analysis.unsupported_ops().items()}
    # All unsupported operators here are elementwise. Count a deliberately high
    # 20 arithmetic operations per output element to bound their omitted cost.
    conservative = FlopCountAnalysis(model, sample).unsupported_ops_warnings(False)

    def elementwise_upper_bound(_inputs, outputs):
        shape = get_shape(outputs[0])
        if shape is None:
            raise ValueError("Cannot bound an unsupported operator with unknown output shape")
        return 20 * prod(shape)

    for operator in ("aten::hardswish_", "aten::hardsigmoid", "aten::mul", "aten::add_"):
        conservative.set_op_handle(operator, elementwise_upper_bound)
    conservative.set_op_handle("aten::dropout_", lambda _inputs, _outputs: 0)
    conservative.set_op_handle("aten::cat", lambda _inputs, _outputs: 0)
    bounded_flops = int(conservative.total())
    remaining = dict(conservative.unsupported_ops())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "input_shape": list(sample.shape), "output_shape": list(output.shape),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_epoch": checkpoint_epoch,
        "parameters": parameters, "fvcore_counted_flops": flops,
        "unsupported_operators": unsupported,
        "conservative_flops_upper_bound": bounded_flops,
        "upper_bound_assumption": "20 operations per output element for listed elementwise operators; eval dropout 0",
        "remaining_unsupported_operators": remaining,
        "parameter_limit": 5_000_000, "flop_limit": 500_000_000,
        "counted_flops_within_limit": flops <= 500_000_000,
        "conservative_bound_within_limit": bounded_flops <= 500_000_000 and not remaining,
        "parameters_within_limit": parameters <= 5_000_000,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".txt":
            report = ["FIQA MobileNetV3-Small model profile", ""]
            report += [f"{key}: {value}" for key, value in result.items()]
            args.output.write_text("\n".join(report) + "\n", encoding="utf-8")
        else:
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if parameters > 5_000_000 or bounded_flops > 500_000_000 or remaining:
        raise SystemExit("FAIL: model exceeds the competition limit")


if __name__ == "__main__":
    main()
