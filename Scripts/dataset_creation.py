import glob
import json
import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def create_dataframe(root_dir: str, label_folders: list, csv_path: str):
    if os.path.exists(csv_path):
        print("\nDataset loaded from memory!")
    else:
        df = pd.DataFrame()
        d = {i: [] for i in label_folders}
        d["Text"] = []
        os.chdir(os.path.join(root_dir, "Sentences/"))
        for _, file_name in tqdm(enumerate(list(glob.glob("*.txt")))):
            with open(file_name) as file:
                lines = file.read().splitlines()
                d["Text"].extend(lines)
                for label_folder in label_folders:
                    label_folder_path = os.path.join(root_dir, label_folder, file_name)
                    with open(label_folder_path) as label_file:
                        d[label_folder].extend(label_file.read().splitlines())

        df["Text"] = d["Text"]
        for label in label_folders:
            df[label.split("_")[1]] = d[label]

        df.to_csv(csv_path, index=False)
        print("\nDataset creation successful")


def split_dataframe(
    csv_path: str, train_path: str, test_path: str, test_size: float, random_state: int
):
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def main():
    if len(sys.argv) != 2:
        print("\nPass Config JSON as argument!")
        print(
            "--------------------\nTrain, test and val dataset creation failed!\n--------------------\n"
        )
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        hyper_params = json.load(fjson)

    label_folders = [
        "Labels_A",
        "Labels_CH",
        "Labels_CR",
        "Labels_J",
        "Labels_LAW",
        "Labels_LTD",
        "Labels_TER",
        "Labels_USE",
    ]

    root_dir = hyper_params["csv"]["rootDir"]
    csv_path = hyper_params["csv"]["dataframePath"]
    train_csv_path = hyper_params["csv"]["trainDataframePath"]
    test_csv_path = hyper_params["csv"]["testDataframePath"]
    val_csv_path = hyper_params["csv"]["valDataframePath"]
    test_size = hyper_params["csv"]["testSize"]
    val_size = hyper_params["csv"]["valSize"]
    random_state = hyper_params["csv"]["randomState"]

    csv_path = os.path.join(root_dir, csv_path)
    train_csv_path = os.path.join(root_dir, train_csv_path)
    test_csv_path = os.path.join(root_dir, test_csv_path)
    val_csv_path = os.path.join(root_dir, val_csv_path)
    test_size = float(test_size)
    val_size = float(val_size)

    create_dataframe(
        root_dir=root_dir,
        label_folders=label_folders,
        csv_path=csv_path,
    )
    split_dataframe(
        csv_path=csv_path,
        train_path=train_csv_path,
        test_path=test_csv_path,
        test_size=test_size,
        random_state=random_state,
    )
    split_dataframe(
        csv_path=train_csv_path,
        train_path=train_csv_path,
        test_path=val_csv_path,
        test_size=val_size,
        random_state=random_state,
    )
    print(
        "--------------------\nTrain, test and val dataset creation successfull!\n--------------------\n"
    )


if __name__ == "__main__":
    main()
