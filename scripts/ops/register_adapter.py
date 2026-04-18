import json
import argparse
import sys
from pathlib import Path

def gate(clip_score: float, min_clip: float = 22.0) -> bool:
    """Verifies if the fine-tuned model passes the semantic retention gate."""
    return clip_score >= min_clip

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a LoRA adapter if it passes the CLIP evaluation gate.")
    parser.add_argument("--adapter", required=True, help="Adapter name (e.g. 'light' or 'overfit')")
    parser.add_argument("--clip_score", type=float, required=True, help="Evaluated CLIP semantic similarity score")
    parser.add_argument("--min_clip", type=float, default=22.0, help="Minimum acceptable CLIP score for production")
    parser.add_argument("--stage", default="staging", help="Deployment stage target (e.g. 'production' or 'staging')")
    args = parser.parse_args()

    record = {
        "adapter": args.adapter,
        "clip_score": args.clip_score,
        "stage": args.stage,
        "registered": False,
        "gate_passed": False
    }

    if not gate(args.clip_score, args.min_clip):
        print(f"GATE FAILED: {args.adapter} clip={args.clip_score} < threshold={args.min_clip}")
        print("This adapter exhibits catastrophic forgetting and will NOT be promoted to production.")
    else:
        print(f"GATE PASSED: {args.adapter} clip={args.clip_score} >= threshold={args.min_clip}")
        record["registered"] = True
        record["gate_passed"] = True
        print(f"REGISTERED: {args.adapter} successfully promoted to {args.stage}!")

    out = Path("eval/adapter_registry.json")
    out.parent.mkdir(exist_ok=True)
    
    # Read existing or create new list
    if out.exists() and out.stat().st_size > 0:
        with open(out, 'r') as f:
            registry = json.load(f)
    else:
        registry = []
        
    registry.append(record)

    out.write_text(json.dumps(registry, indent=2))
