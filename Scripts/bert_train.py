import os
import sys
import json
import pytorch_lightning as pl
from bert_utils import create_dataframe, get_stat_details, Model, ToSDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import BertTokenizerFast as BertTokenizer


def main():
    if len(sys.argv) != 2:
        print("\nPass Config JSON as argument!")
        print("--------------------\nModel training failed!\n--------------------\n")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        hyper_params = json.load(fjson)

    root_dir = hyper_params["csv"]["rootDir"]
    train_csv_path = hyper_params["csv"]["trainDataframePath"]
    test_csv_path = hyper_params["csv"]["testDataframePath"]
    val_csv_path = hyper_params["csv"]["valDataframePath"]

    train_csv_path = os.path.join(root_dir, train_csv_path)
    test_csv_path = os.path.join(root_dir, test_csv_path)
    val_csv_path = os.path.join(root_dir, val_csv_path)

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

    else:
        stage = "Stage2"
        stage_train_csv_path = hyper_params["knowledgedistillation"][
            "stage2trainDataframePath"
        ]
        stage_test_csv_path = hyper_params["knowledgedistillation"][
            "stage2testDataframePath"
        ]
        stage_val_csv_path = hyper_params["knowledgedistillation"][
            "stage2valDataframePath"
        ]

    stage_train_csv_path = os.path.join(root_dir, stage_train_csv_path)
    stage_test_csv_path = os.path.join(root_dir, stage_test_csv_path)
    stage_val_csv_path = os.path.join(root_dir, stage_val_csv_path)

    model_name = hyper_params["train"]["modelName"]
    label_column = hyper_params["train"]["labelColumn"]
    batch_size = hyper_params["train"]["batchSize"]
    token_length = hyper_params["train"]["tokenLength"]
    epochs = hyper_params["train"]["noEpochs"]
    learning_rate = hyper_params["train"]["learningRate"]
    classes = hyper_params["train"]["noClasses"]
    checkpoint_dir = hyper_params["train"]["checkpointDir"]
    checkpoint_filename = hyper_params["train"]["checkpointFileName"]
    save_top_k = hyper_params["train"]["saveTopK"]
    verbose_flag = hyper_params["train"]["verboseFlag"]
    checkpoint_monitor = hyper_params["train"]["checkpointMonitor"]
    checkpoint_mode = hyper_params["train"]["checkpointMode"]
    paitence_value = hyper_params["train"]["paitenceValue"]
    gpu_count = hyper_params["train"]["gpuCount"]
    refresh_rate = hyper_params["train"]["refreshRate"]

    checkpoint_dir = os.path.join(root_dir, checkpoint_dir)

    train_df, test_df, val_df = create_dataframe(
        stage=stage,
        stage_train_csv_path=stage_train_csv_path,
        stage_test_csv_path=stage_test_csv_path,
        stage_val_csv_path=stage_val_csv_path,
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        val_csv_path=val_csv_path,
    )

    if stage == "Stage1":
        get_stat_details(
            dataframe=train_df,
            root_dir=root_dir,
            title="{}TrainDistribution".format(stage),
        )
        get_stat_details(
            dataframe=test_df,
            root_dir=root_dir,
            title="{}TestDistribution".format(stage),
        )
        get_stat_details(
            dataframe=val_df, root_dir=root_dir, title="{}ValDistribution".format(stage)
        )

    tokenizer = BertTokenizer.from_pretrained(model_name)

    data_module = ToSDataModule(
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        tokenizer=tokenizer,
        label_column=label_column,
        batch_size=batch_size,
        max_token_len=token_length,
        stage_flag=stage_flag,
    )

    steps_per_epoch = len(train_df) // batch_size
    total_training_steps = steps_per_epoch * epochs
    warmup_steps = total_training_steps // 5

    model = Model(
        model_name=model_name,
        learning_rate=learning_rate,
        n_classes=classes,
        n_training_steps=total_training_steps,
        n_warmup_steps=warmup_steps,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=checkpoint_filename,
        save_top_k=save_top_k,
        verbose=verbose_flag,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
    )

    early_stopping_callback = EarlyStopping(
        monitor=checkpoint_monitor, patience=paitence_value
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=epochs,
        gpus=gpu_count,
        progress_bar_refresh_rate=refresh_rate,
    )

    trainer.fit(model, data_module)

    print(
        "--------------------\n{} training successfull!\n--------------------\n".format(
            stage
        )
    )


if __name__ == "__main__":
    main()
