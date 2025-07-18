import os
import shutil
from roboflow import Roboflow
import time
import yaml

def find_data_yaml(directory):
    for root, _, files in os.walk(directory):
        if "data.yaml" in files:
            return os.path.join(root, "data.yaml")
    return None

def download_dataset(api_key, workspace_id, project_id, version_number, download_path, dataset_name):
    try:
        print(f"\n=== Downloading {dataset_name} ===")
        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace_id)
        project = workspace.project(project_id)

        version = project.version(version_number)
        dataset = version.download("yolov8", location=download_path)
        dataset_folder = dataset.location

        yaml_path = find_data_yaml(dataset_folder)
        if yaml_path:
            print(f"Success: {dataset_name} data.yaml found at {yaml_path}")
            return yaml_path, dataset_folder
        else:
            print(f"Error: data.yaml not found in {dataset_folder}")
            return None, None

    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        return None, None

def combine_datasets(datasets, combined_path, class_name="person"):
    os.makedirs(combined_path, exist_ok=True)
    splits = ["train", "valid", "test"]
    for split in splits:
        os.makedirs(f"{combined_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{combined_path}/{split}/labels", exist_ok=True)

    for dataset in datasets:
        dataset_name = dataset["dataset_name"]
        dataset_folder = dataset["dataset_folder"]
        if not dataset_folder:
            print(f"Skipping {dataset_name} due to missing data.")
            continue

        for split in splits:
            src_img_dir = f"{dataset_folder}/{split}/images"
            src_lbl_dir = f"{dataset_folder}/{split}/labels"

            if not os.path.exists(src_img_dir):
                print(f"Split '{split}' not found in {dataset_name}, skipping.")
                continue

            for img_file in os.listdir(src_img_dir):
                if img_file.endswith((".jpg", ".png", ".jpeg")):
                    base_name = os.path.splitext(img_file)[0]
                    src_img = os.path.join(src_img_dir, img_file)
                    src_lbl = os.path.join(src_lbl_dir, f"{base_name}.txt")

                    dst_img = os.path.join(combined_path, split, "images", f"{dataset_name}_{img_file}")
                    dst_lbl = os.path.join(combined_path, split, "labels", f"{dataset_name}_{base_name}.txt")

                    shutil.copy(src_img, dst_img)

                    if os.path.exists(src_lbl):
                        with open(src_lbl) as f:
                            lines = f.readlines()
                        with open(dst_lbl, "w") as f:
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) == 5:
                                    parts[0] = "0"
                                    f.write(" ".join(parts) + "\n")

    data_yaml = {
        "train": f"{combined_path}/train/images",
        "val": f"{combined_path}/valid/images",
        "test": f"{combined_path}/test/images",
        "nc": 1,
        "names": [class_name]
    }
    with open(f"{combined_path}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    print(f"\nCombined dataset created at {combined_path}")

if __name__ == "__main__":
    api_key = "API key"

    base_download_path = "/Users/anton/PycharmProjects/100WinnerTracking/RunningDatasets"
    combined_path = f"{base_download_path}/combined_running_dataset"

    datasets = [
        {
            "workspace_id": "gg-3i5yz",
            "project_id": "running-cefql",
            "version_number": 1,
            "download_path": f"{base_download_path}/running_cefql",
            "dataset_name": "running_cefql"
        },
        {
            "workspace_id": "anton-mre5k",
            "project_id": "human-running-t6vec-w4srv",
            "version_number": 1,  # cannot find, then skip
            "download_path": f"{base_download_path}/human_running_t6vec",
            "dataset_name": "human_running_t6vec"
        },
        {
            "workspace_id": "kmutt-ai",
            "project_id": "running-muumz",
            "version_number": 1,
            "download_path": f"{base_download_path}/running_muumz",
            "dataset_name": "running_muumz"
        }
    ]

    successful_downloads = 0
    for dataset in datasets:
        start_time = time.time()
        yaml_path, dataset_folder = download_dataset(
            api_key,
            dataset["workspace_id"],
            dataset["project_id"],
            dataset["version_number"],
            dataset["download_path"],
            dataset["dataset_name"]
        )
        if yaml_path:
            successful_downloads += 1
            dataset["yaml_path"] = yaml_path
            dataset["dataset_folder"] = dataset_folder
        else:
            dataset["yaml_path"] = None
            dataset["dataset_folder"] = None
        print(f"Time taken: {time.time() - start_time:.2f}s")

    if successful_downloads > 0:
        combine_datasets(datasets, combined_path)
    else:
        print("No datasets downloaded successfully.")
