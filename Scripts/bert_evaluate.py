import os
import sys
import json
import torch
import pandas as pd
from bert_utils import create_kd_dataframe, generate_report, Model
from transformers import BertTokenizerFast as BertTokenizer


def main():
    if len(sys.argv) != 2:
        print("\nPass Config JSON as argument!")
        print("--------------------\nModel evaluation failed!\n--------------------\n")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        hyper_params = json.load(fjson)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--------------------\nGPU Available!\n--------------------\n")
    else:
        device = torch.device("cpu")
        print("--------------------\nGPU Not Available!\n--------------------\n")

    stage_flag = hyper_params["knowledgedistillation"]["stageFlag"]
    if stage_flag == "1":
        stage = "Stage1"
        stage_train_csv_path = hyper_params["knowledgedistillation"][
            "stage1trainDataframePath"
        ]
        stage_test_csv_path = hyper_params["knowledgedistillation"][
            "stage1testDataframePath"
        ]
        stage_val_csv_path = hyper_params["knowledgedistillation"][
            "stage1valDataframePath"
        ]

        kd_train_csv_path = hyper_params["knowledgedistillation"][
            "stage1kdtrainDataframePath"
        ]
        kd_test_csv_path = hyper_params["knowledgedistillation"][
            "stage1kdtestDataframePath"
        ]
        kd_val_csv_path = hyper_params["knowledgedistillation"][
            "stage1kdvalDataframePath"
        ]

    else:
        stage = "Stage2"
        stage_train_csv_path = hyper_params["knowledgedistillation"][
            "stage1trainDataframePath"
        ]
        stage_test_csv_path = hyper_params["knowledgedistillation"][
            "stage1testDataframePath"
        ]
        stage_val_csv_path = hyper_params["knowledgedistillation"][
            "stage1valDataframePath"
        ]

        kd_train_csv_path = hyper_params["knowledgedistillation"][
            "stage2kdtrainDataframePath"
        ]
        kd_test_csv_path = hyper_params["knowledgedistillation"][
            "stage2kdtestDataframePath"
        ]
        kd_val_csv_path = hyper_params["knowledgedistillation"][
            "stage2kdvalDataframePath"
        ]

    root_dir = hyper_params["csv"]["rootDir"]
    checkpoint_filename = hyper_params["evaluate"]["checkpointFileName"]
    train_metrics_filename = hyper_params["evaluate"]["trainMetricsFileName"]
    test_metrics_filename = hyper_params["evaluate"]["testMetricsFileName"]
    val_metrics_filename = hyper_params["evaluate"]["valMetricsFileName"]

    model_name = hyper_params["evaluate"]["modelName"]
    token_length = hyper_params["evaluate"]["tokenLength"]
    classes = hyper_params["evaluate"]["noClasses"]
    label_column = hyper_params["evaluate"]["labelColumn"]

    threshold = hyper_params["evaluate"]["thresholdValue"]
    metrics_label = hyper_params["evaluate"]["metricsLabel"]

    stage_train_csv_path = os.path.join(root_dir, stage_train_csv_path)
    stage_test_csv_path = os.path.join(root_dir, stage_test_csv_path)
    stage_val_csv_path = os.path.join(root_dir, stage_val_csv_path)
    kd_train_csv_path = os.path.join(root_dir, kd_train_csv_path)
    kd_test_csv_path = os.path.join(root_dir, kd_test_csv_path)
    kd_val_csv_path = os.path.join(root_dir, kd_val_csv_path)

    checkpoint_filepath = os.path.join(root_dir, checkpoint_filename)
    train_metrics_filepath = os.path.join(root_dir, train_metrics_filename)
    test_metrics_filepath = os.path.join(root_dir, test_metrics_filename)
    val_metrics_filepath = os.path.join(root_dir, val_metrics_filename)

    train_df = pd.read_csv(stage_train_csv_path)
    test_df = pd.read_csv(stage_test_csv_path)
    val_df = pd.read_csv(stage_val_csv_path)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = Model.load_from_checkpoint(
        model_name=model_name, n_classes=classes, checkpoint_path=checkpoint_filepath
    )

    generate_report(
        model=model,
        device=device,
        dataframe=train_df,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=token_length,
        stage_flag=stage_flag,
        threshold=threshold,
        metrics_label=metrics_label,
        metrics_filepath=train_metrics_filepath,
    )
    generate_report(
        model=model,
        device=device,
        dataframe=test_df,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=token_length,
        stage_flag=stage_flag,
        threshold=threshold,
        metrics_label=metrics_label,
        metrics_filepath=test_metrics_filepath,
    )
    generate_report(
        model=model,
        device=device,
        dataframe=val_df,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=token_length,
        stage_flag=stage_flag,
        threshold=threshold,
        metrics_label=metrics_label,
        metrics_filepath=val_metrics_filepath,
    )
    create_kd_dataframe(
        model,
        device,
        dataframe=train_df,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=token_length,
        stage_flag=stage_flag,
        new_csv_path=kd_train_csv_path,
    )
    create_kd_dataframe(
        model,
        device,
        dataframe=test_df,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=token_length,
        stage_flag=stage_flag,
        new_csv_path=kd_test_csv_path,
    )
    create_kd_dataframe(
        model,
        device,
        dataframe=val_df,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=token_length,
        stage_flag=stage_flag,
        new_csv_path=kd_val_csv_path,
    )

    print(
        "--------------------\n{} evaluation successfull!\n--------------------\n".format(
            stage
        )
    )


if __name__ == "__main__":
    main()
