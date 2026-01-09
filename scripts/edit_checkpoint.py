import shutil
import sys
import os
import torch
import argparse
import re

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def edit_checkpoint(checkpoint_path):
    print("# Edit PyTorch Checkpoint")
    print(
        "This script loads a PyTorch checkpoint, modifies keys from *.stages.x.* to *.stages_x.*, and saves the updated checkpoint."
    )

    print("\n## Load the checkpoint")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("\n## Modify the keys")
    modified_keys = {}
    pattern = re.compile(r"(.*\.stages)\.(\_?)(\d+)(\..*)")  # _ is optional

    for old_key in list(checkpoint["state_dict"].keys()):
        match = pattern.match(old_key)
        if match:
            new_key = f"{match.group(1)}_{match.group(3)}{match.group(4)}"
            modified_keys[old_key] = new_key

    for old_key, new_key in modified_keys.items():
        checkpoint["state_dict"][new_key] = checkpoint["state_dict"].pop(
            old_key
        )
        print(f"Key '{old_key}' has been changed to '{new_key}'")

    if not modified_keys:
        print(
            "No keys matching the pattern *.stages.x.* were found in the checkpoint"
        )

    print("\n## Save the updated checkpoint")
    dir_name, file_name = os.path.split(checkpoint_path)
    new_file_name = "updated_" + file_name
    new_checkpoint_path = os.path.join(dir_name, new_file_name)

    torch.save(checkpoint, new_checkpoint_path)
    print(f"Updated checkpoint saved to: {new_checkpoint_path}")

    print("\n## Verify the changes")
    updated_checkpoint = torch.load(new_checkpoint_path, map_location="cpu")

    all_correct = True
    for old_key, new_key in modified_keys.items():
        if (
            new_key in updated_checkpoint["state_dict"]
            and old_key not in updated_checkpoint["state_dict"]
        ):
            print(
                f"Verification successful: '{old_key}' has been updated to '{new_key}'"
            )
        else:
            print(
                f"Verification failed: '{old_key}' to '{new_key}' update was not successful"
            )
            all_correct = False

    if all_correct:
        print("All key updates were successful")
        # Create a backup of the original checkpoint
        backup_file_name = file_name + ".backup"
        backup_checkpoint_path = os.path.join(dir_name, backup_file_name)
        shutil.copy2(checkpoint_path, backup_checkpoint_path)
        print(f"Original checkpoint backed up to: {backup_checkpoint_path}")

        # Rename the new checkpoint to the original name
        os.rename(new_checkpoint_path, checkpoint_path)
        print(
            f"Updated checkpoint saved with original name: {checkpoint_path}"
        )

        # Update the path for verification
        new_checkpoint_path = checkpoint_path
    else:
        print("Some key updates failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edit a PyTorch checkpoint file."
    )
    parser.add_argument(
        "checkpoint_path", help="Path to the checkpoint file to be edited."
    )
    args = parser.parse_args()

    edit_checkpoint(args.checkpoint_path)
