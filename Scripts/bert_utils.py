import ast
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from collections import Counter
from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import classification_report
from torch.nn import BCELoss, Linear
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

pl.seed_everything(42)


def create_isUnfair(dataframe):
    isUnfair = []
    for idx in range(len(dataframe)):
        labels = ast.literal_eval(dataframe.iloc[idx]["Labels"].replace(" ", ", "))
        arr = np.array(labels)
        if np.sum(arr) > 0:
            isUnfair.append(1)
        else:
            isUnfair.append(0)
    dataframe["isUnfair"] = isUnfair
    return dataframe


def get_stat_details(dataframe, root_dir: str, title: str):
    image_filename = "Images/" + title + ".png"
    image_filepath = os.path.join(root_dir, image_filename)
    distribution = Counter(dataframe["isUnfair"])
    fair = distribution[0]
    unfair = distribution[1]
    total = fair + unfair

    print("--------------------\n{}".format(title))
    print("Total:", total)
    print("Unfair:", unfair)
    print("Fair:", fair)

    explode = [0.2, 0]
    labels = ["Unfair", "Fair"]
    shadow = True
    unfair = round(((unfair / total) * 100), 2)
    fair = round(((fair / total) * 100), 2)

    data = [unfair, fair]
    plt.pie(data, labels=labels, explode=explode, autopct="%d%%", shadow=shadow)
    plt.legend(title=title)
    plt.savefig(image_filepath)
    plt.show()


def create_dataframe(
    stage: str,
    stage_train_csv_path: str,
    stage_test_csv_path: str,
    stage_val_csv_path: str,
    train_csv_path: str,
    test_csv_path: str,
    val_csv_path: str,
):
    try:
        train_df = pd.read_csv(stage_train_csv_path)
        test_df = pd.read_csv(stage_test_csv_path)
        val_df = pd.read_csv(stage_val_csv_path)
        print("\n{} train, test and val dataframes loaded from memory!".format(stage))

    except:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        val_df = pd.read_csv(val_csv_path)

        train_df = train_df.dropna(axis=0, how="any")
        test_df = test_df.dropna(axis=0, how="any")
        val_df = val_df.dropna(axis=0, how="any")
        train_df = create_isUnfair(train_df)
        test_df = create_isUnfair(test_df)
        val_df = create_isUnfair(val_df)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        if stage == "Stage1":
            train_df.to_csv(stage_train_csv_path, index=False)
            test_df.to_csv(stage_test_csv_path, index=False)
            val_df.to_csv(stage_val_csv_path, index=False)
        else:
            train_df = train_df[train_df["isUnfair"] == 1]
            test_df = test_df[test_df["isUnfair"] == 1]
            val_df = val_df[val_df["isUnfair"] == 1]
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            train_df.to_csv(stage_train_csv_path, index=False)
            test_df.to_csv(stage_test_csv_path, index=False)
            val_df.to_csv(stage_val_csv_path, index=False)
        print("\nCreated {} train, test and val dataframes!".format(stage))
    return train_df, test_df, val_df


class ToSDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        label_column: str,
        max_token_len: int,
        stage_flag: str,
    ):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.label_column = label_column.split(",")
        self.max_token_len = max_token_len
        self.stage_flag = stage_flag
        if self.stage_flag == "1":
            self.label_column = self.label_column[0]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        text = self.dataframe.iloc[index]["Text"]
        if self.stage_flag == "1":
            labels = [self.dataframe.iloc[index][self.label_column]]
        else:
            labels = self.dataframe.iloc[index][self.label_column]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels),
        )


class ToSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        test_df,
        val_df,
        tokenizer,
        label_column: str,
        batch_size: int,
        max_token_len: int,
        stage_flag: str,
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.label_column = label_column
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.stage_flag = stage_flag

    def setup(self, stage=None):
        self.train_dataset = ToSDataset(
            dataframe=self.train_df,
            tokenizer=self.tokenizer,
            label_column=self.label_column,
            max_token_len=self.max_token_len,
            stage_flag=self.stage_flag,
        )
        self.test_dataset = ToSDataset(
            dataframe=self.test_df,
            tokenizer=self.tokenizer,
            label_column=self.label_column,
            max_token_len=self.max_token_len,
            stage_flag=self.stage_flag,
        )
        self.val_dataset = ToSDataset(
            dataframe=self.val_df,
            tokenizer=self.tokenizer,
            label_column=self.label_column,
            max_token_len=self.max_token_len,
            stage_flag=self.stage_flag,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, num_workers=8
        )


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        learning_rate=2e-5,
        n_training_steps=None,
        n_warmup_steps=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.criterion = BCELoss()
        self.model = BertModel.from_pretrained(self.model_name, return_dict=True)
        self.classifier = Linear(self.model.config.hidden_size, n_classes)
        self.learning_rate = learning_rate
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, _ = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )


def predict(
    model,
    device,
    dataframe,
    tokenizer,
    label_column: str,
    max_token_len: int,
    stage_flag: str,
):
    dataset = ToSDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=max_token_len,
        stage_flag=stage_flag,
    )
    predictions = []
    labels = []
    text = []
    for item in dataset:
        _, prediction = model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device),
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())
        text.append(item["text"])

    return predictions, labels, text


def generate_report(
    model,
    device,
    dataframe,
    tokenizer,
    label_column: str,
    max_token_len: int,
    stage_flag: str,
    threshold: float,
    metrics_label: str,
    metrics_filepath: str,
):

    model.eval()
    model.freeze()
    model = model.to(device)

    predictions, labels, _ = predict(
        model=model,
        device=device,
        dataframe=dataframe,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=max_token_len,
        stage_flag=stage_flag,
    )
    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()

    accuracy_value = accuracy(predictions, labels, threshold=threshold).item()
    accuracy_value = round(accuracy_value, 3)

    metrics_label = metrics_label.split(",")
    y_pred = predictions.numpy()
    y_true = labels.numpy()
    upper, lower = 1, 0
    y_pred = np.where(y_pred > threshold, upper, lower)
    report = classification_report(
        y_true, y_pred, target_names=metrics_label, digits=3, zero_division=0
    )

    with open(metrics_filepath, "w") as metrics_file:
        metrics_file.write(
            "--------------------\nAccuracy: {}\n--------------------\n\n--------------------\nClassification Report\n\n{}\n--------------------\n".format(
                accuracy_value, report
            )
        )


def create_kd_dataframe(
    model,
    device,
    dataframe,
    tokenizer,
    label_column: str,
    max_token_len: int,
    stage_flag: str,
    new_csv_path: str,
):
    kd_dataframe = pd.DataFrame()
    predictions, labels, text = predict(
        model=model,
        device=device,
        dataframe=dataframe,
        tokenizer=tokenizer,
        label_column=label_column,
        max_token_len=max_token_len,
        stage_flag=stage_flag,
    )
    kd_dataframe["Text"] = text
    if stage_flag == "1":
        kd_dataframe["isUnfair"] = [i.item() for i in labels]
        kd_dataframe["BERT_Predicted"] = [i.item() for i in predictions]
    else:
        kd_dataframe["Labels"] = [list(i.numpy()) for i in labels]
        kd_dataframe["BERT_Predicted"] = [list(i.cpu().numpy()) for i in predictions]
    kd_dataframe.to_csv(new_csv_path, index=False)
    return kd_dataframe
