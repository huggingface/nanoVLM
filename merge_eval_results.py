import argparse
import glob
import json
import os


def merge_results():
    parser = argparse.ArgumentParser(description="Merge lmms-eval results for a specific step.")
    parser.add_argument("--run_name", type=str, required=True, help="The name of the training run.")
    parser.add_argument("--global_step", type=int, required=True, help="Global step for which to merge results.")
    args = parser.parse_args()

    output_dir = os.path.join("eval_results", args.run_name)
    if not os.path.exists(output_dir):
        print(f"Directory not found: {output_dir}")
        return

    merged_data = {"global_step": args.global_step, "results": {}}
    files_to_merge = glob.glob(os.path.join(output_dir, f"step_{args.global_step}_*.json"))

    if not files_to_merge:
        print(f"No partial result files found for step {args.global_step} in {output_dir}")
        return

    print(f"Merging {len(files_to_merge)} files for step {args.global_step}...")

    for file_path in files_to_merge:
        with open(file_path) as f:
            data = json.load(f)
            if "results" in data:
                merged_data["results"].update(data["results"])
        os.remove(file_path)

    merged_file_path = os.path.join(output_dir, f"step_{args.global_step}.json")
    with open(merged_file_path, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged results saved to {merged_file_path}")


if __name__ == "__main__":
    merge_results()
