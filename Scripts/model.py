import os
import sys
import json
import numpy as np
import pandas as pd
from model_utils import *
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    if len(sys.argv) != 2:
        print("\nPass Config JSON as argument!")
        print("--------------------\nModel training failed!\n--------------------\n")
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        hyper_params = json.load(fjson)

    root_dir = hyper_params["csv"]["rootDir"]
    stage_flag = hyper_params["train"]["stageFlag"]
    if stage_flag == "Stage1":
        stage = "Stage1"
        train_csv_path = hyper_params["csv"]["stage1trainDataframePath"]
        test_csv_path = hyper_params["csv"]["stage1testDataframePath"]
        val_csv_path = hyper_params["csv"]["stage1valDataframePath"]
    else:
        stage = "Stage2"
        train_csv_path = hyper_params["csv"]["stage2trainDataframePath"]
        test_csv_path = hyper_params["csv"]["stage2testDataframePath"]
        val_csv_path = hyper_params["csv"]["stage2valDataframePath"]

    fasttext_filename = hyper_params["vectors"]["fasttextFileName"]
    embedding_matrix_filename = hyper_params["vectors"]["embeddingmatrixFileName"]
    tokenizer_filename = hyper_params["vectors"]["tokenizerFileName"]
    word_dict_filename = hyper_params["vectors"]["dictFileName"]
    checkpoint_filename = hyper_params["train"]["checkpointFileName"]

    model_flag = hyper_params["train"]["modelFlag"]
    bidirectional_flag = hyper_params["train"]["bidirectionalFlag"]

    oov_token = hyper_params["train"]["oovToken"]
    pad_token = hyper_params["train"]["padToken"]
    embed_dim = hyper_params["train"]["embedDim"]
    model_units = hyper_params["train"]["modelUnits"]
    hidden_units = hyper_params["train"]["hiddenUnits"]
    classes = hyper_params["train"]["noClasses"]
    dropout = hyper_params["train"]["dropoutValue"]
    epochs = hyper_params["train"]["noEpochs"]
    batch_size = hyper_params["train"]["batchSize"]
    threshold = hyper_params["train"]["thresholdValue"]
    checkpoint_monitor = hyper_params["train"]["checkpointMonitor"]
    checkpoint_mode = hyper_params["train"]["checkpointMode"]
    paitence_value = hyper_params["train"]["paitenceValue"]
    image_filename = hyper_params["train"]["imageFileName"]

    metrics_label = hyper_params["evaluate"]["metricsLabel"]
    metrics_filename = hyper_params["evaluate"]["metricsFileName"]

    train_csv_path = os.path.join(root_dir, train_csv_path)
    test_csv_path = os.path.join(root_dir, test_csv_path)
    val_csv_path = os.path.join(root_dir, val_csv_path)

    fasttext_path = os.path.join(root_dir, fasttext_filename)
    checkpoint_path = os.path.join(root_dir, checkpoint_filename)
    embedding_matrix_path = os.path.join(root_dir, embedding_matrix_filename)
    tokenizer_path = os.path.join(root_dir, tokenizer_filename)
    word_dict_path = os.path.join(root_dir, word_dict_filename)
    image_path = os.path.join(root_dir, image_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    val_df = pd.read_csv(val_csv_path)

    tokenizer, word_dict = load_tokenizer_word_dict(
        dataframe=train_df,
        tokenizer_path=tokenizer_path,
        word_dict_path=word_dict_path,
        oov_token=oov_token,
        pad_token=pad_token,
    )
    vocab_size = len(word_dict) + 1
    embedding_matrix = load_embedding_matrix(
        word_dict=word_dict,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        embedding_matrix_path=embedding_matrix_path,
        fasttext_path=fasttext_path,
    )

    train_sequences = tokenizer.texts_to_sequences(train_df["Text"])
    test_sequences = tokenizer.texts_to_sequences(test_df["Text"])
    val_sequences = tokenizer.texts_to_sequences(val_df["Text"])

    X_train = pad_sequences(train_sequences, maxlen=sequence_length, padding="pre")
    X_test = pad_sequences(test_sequences, maxlen=sequence_length, padding="pre")
    X_val = pad_sequences(val_sequences, maxlen=sequence_length, padding="pre")

    sequence_length = hyper_params["train"].get("sequenceLength", "")

    if sequence_length == "":
        sequence_length = [len(data) for data in train_sequences]
        sequence_length = max(sequence_length)

    if stage == "Stage1":
        y_train = np.array(train_df["isUnfair"])
        y_test = np.array(test_df["isUnfair"])
        y_val = np.array(val_df["isUnfair"])
    else:
        y_train = np.array(list(train_df["Labels"]))
        y_test = np.array(list(test_df["Labels"]))
        y_val = np.array(list(val_df["Labels"]))

    model = Model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        word_dict=word_dict,
        embed_dim=embed_dim,
        sequence_length=sequence_length,
        embedding_matrix=embedding_matrix,
        stage_flag=stage_flag,
        model_flag=model_flag,
        bidirectional_flag=bidirectional_flag,
        model_units=model_units,
        hidden_units=hidden_units,
        classes=classes,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        threshold=threshold,
        checkpoint_path=checkpoint_path,
        monitor=checkpoint_monitor,
        paitence_value=paitence_value,
        mode=checkpoint_mode,
        metrics_label=metrics_label,
        metrics_path=metrics_path,
        image_path=image_path,
    )

    model.train()
    model.metrics()

    print(
        "--------------------\n{} training and evaluation successfull!\n--------------------\n".format(
            stage
        )
    )


if __name__ == "__main__":
    main()
