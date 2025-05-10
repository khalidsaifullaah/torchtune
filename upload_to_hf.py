import argparse
import os
from huggingface_hub import HfApi, create_repo, whoami

def main():
    parser = argparse.ArgumentParser(description="Upload a trained model to Hugging Face Hub.")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the trained model directory (e.g., /tmp/torchtune/llama3_2_3B/lora_single_device/epoch_0)"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="tomg-group-umd/test",
        help="Name of the Hugging Face Hub repository."
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of the repository."
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Whether to create a Pull Request instead of pushing directly."
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="",
        help="Path inside the repository where files will be uploaded."
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload model",
        help="Commit message for the upload."
    )
    args = parser.parse_args()

    # Ensure the model directory exists
    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(f"The specified model directory does not exist: {args.model_dir}")

    # Initialize Hugging Face API
    api = HfApi()

    # Get the current user's username
    username = whoami()["name"]
    repo_id = f"{username}/{args.repo_name}"

    # Create the repository if it doesn't exist
    try:
        create_repo(repo_id, repo_type=args.repo_type, exist_ok=True)
        print(f"Repository '{repo_id}' is ready.")
    except Exception as e:
        print(f"Failed to create or access the repository: {e}")
        return

    # Upload the folder to the repository
    try:
        api.upload_folder(
            folder_path=args.model_dir,
            repo_id=repo_id,
            repo_type=args.repo_type,
            path_in_repo=args.path_in_repo,
            create_pr=args.create_pr,
            commit_message=args.commit_message,
            # private=True
        )
        print(f"Model uploaded successfully to '{repo_id}'.")
        api.update_repo_settings(repo_id=repo_id, private=True)
        print("Should now be private....")
    except Exception as e:
        print(f"Failed to upload the model: {e}")

if __name__ == "__main__":
    main()


