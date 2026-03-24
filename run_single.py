"""
CLI wrapper for single-query PaperVizAgent execution.
Usage:
  python run_single.py --method "method text" --caption "figure caption" [options]
  python run_single.py --input input.json [options]
"""

import asyncio
import json
import argparse
import base64
from pathlib import Path
from datetime import datetime

from agents.vanilla_agent import VanillaAgent
from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.polish_agent import PolishAgent

from utils import config
from utils.paperviz_processor import PaperVizProcessor


async def main():
    parser = argparse.ArgumentParser(description="PaperVizAgent single-query CLI")

    # Input: either --method/--caption or --input file
    parser.add_argument("--method", type=str, default="", help="Method section text")
    parser.add_argument("--caption", type=str, default="", help="Figure caption")
    parser.add_argument("--input", type=str, default="", help="Path to input JSON file with 'method' and 'caption' fields")

    # Options
    parser.add_argument("--task_name", type=str, default="diagram", choices=["diagram", "plot"])
    parser.add_argument("--exp_mode", type=str, default="dev_planner_critic",
                        help="Pipeline mode (dev_planner_critic or dev_full)")
    parser.add_argument("--retrieval_setting", type=str, default="none",
                        choices=["auto", "manual", "random", "none"])
    parser.add_argument("--max_critic_rounds", type=int, default=3)
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates to generate")
    parser.add_argument("--aspect_ratio", type=str, default="16:9")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory (default: results/cli/)")

    args = parser.parse_args()

    # Resolve input
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        method_text = input_data.get("method", input_data.get("content", ""))
        caption_text = input_data.get("caption", "")
    elif args.method and args.caption:
        method_text = args.method
        caption_text = args.caption
    else:
        parser.error("Provide either --method and --caption, or --input <file.json>")
        return

    work_dir = Path(__file__).parent

    exp_config = config.ExpConfig(
        dataset_name="CLI",
        split_name="cli",
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting,
        max_critic_rounds=args.max_critic_rounds,
        model_name=args.model_name,
        work_dir=work_dir,
    )

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    # Build input data list
    data_list = []
    for i in range(args.num_candidates):
        data_list.append({
            "filename": f"cli_input_candidate_{i}",
            "caption": caption_text,
            "content": method_text,
            "visual_intent": caption_text,
            "additional_info": {"rounded_ratio": args.aspect_ratio},
            "max_critic_rounds": args.max_critic_rounds,
            "candidate_id": i,
        })

    print(f"Generating {args.num_candidates} candidate(s) with pipeline: {args.exp_mode}")
    print(f"Retrieval: {args.retrieval_setting} | Critic rounds: {args.max_critic_rounds}")

    results = []
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=min(args.num_candidates, 10), do_eval=False
    ):
        results.append(result_data)
        print(f"  Candidate {len(results)}/{args.num_candidates} done")

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else work_dir / "results" / "cli"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"result_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8", errors="surrogateescape") as f:
        json_string = json.dumps(results, ensure_ascii=False, indent=4)
        json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
        f.write(json_string)

    # Extract and save images
    task_name = args.task_name
    saved_images = []
    for idx, result in enumerate(results):
        # Find final image (last critic round > stylist > planner)
        image_b64 = None
        for round_idx in range(args.max_critic_rounds, -1, -1):
            key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
            if key in result and result[key]:
                image_b64 = result[key]
                break
        if not image_b64:
            for fallback_key in [
                f"target_{task_name}_stylist_desc0_base64_jpg",
                f"target_{task_name}_desc0_base64_jpg",
            ]:
                if fallback_key in result and result[fallback_key]:
                    image_b64 = result[fallback_key]
                    break

        if image_b64:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_bytes = base64.b64decode(image_b64)
            img_path = output_dir / f"candidate_{idx}_{timestamp}.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            saved_images.append(str(img_path))

    print(f"\nDone! Results saved to: {output_dir}")
    print(f"  JSON: {json_path}")
    for img in saved_images:
        print(f"  Image: {img}")


if __name__ == "__main__":
    asyncio.run(main())
