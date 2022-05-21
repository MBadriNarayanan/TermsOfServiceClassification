import os
import sys
import inflect
import json
import nltk
import re
import string
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords

nltk.download("stopwords")
tqdm.pandas()


def check_digit(text, number_name):
    if text.isdigit():
        return number_name.number_to_words(text)
    else:
        return text


def clean_text(text: str, punctuation, stop_words, number_name):
    text = text.lower()
    text = re.sub(r"\-lrb\-", "", text)
    text = re.sub(r"\-rrb\-", "", text)
    text = "".join([i for i in text if i not in punctuation])
    words = nltk.tokenize.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [check_digit(word, number_name) for word in words]
    text = " ".join(words)
    text = re.sub("\W+", " ", text)
    text = re.sub("\s\s+", " ", text)
    return " ".join(words)


def encode(row, columns):
    for heading in columns:
        if row[heading] == -1:
            row[heading] = 0
        else:
            row[heading] = 1
    return row


def clean_dataframe(csv_path, new_csv_path, punctuation, stop_words, number_name):
    df = pd.read_csv(csv_path)
    columns = list(df.columns[1:])
    df["Text"] = df["Text"].progress_apply(
        lambda row: clean_text(row, punctuation, stop_words, number_name)
    )
    df = df.apply(lambda row: encode(row, columns), axis=1)
    df["Labels"] = list(df[columns].values)
    df.to_csv(new_csv_path, index=False)
    return df


def main():
    if len(sys.argv) != 2:
        print("\nPass Config JSON as argument!")
        print(
            "--------------------\nTrain, test and val dataset cleaning failed!\n--------------------\n"
        )
        sys.exit()

    filename = sys.argv[1]
    with open(filename, "rt") as fjson:
        hyper_params = json.load(fjson)

    punctuation = string.punctuation
    stop_words = stopwords.words("english")
    number_name = inflect.engine()

    root_dir = hyper_params["csv"]["rootDir"]
    train_csv_path = hyper_params["csv"]["trainDataframePath"]
    test_csv_path = hyper_params["csv"]["testDataframePath"]
    val_csv_path = hyper_params["csv"]["valDataframePath"]
    clean_train_csv_path = hyper_params["csv"]["cleantrainDataframePath"]
    clean_test_csv_path = hyper_params["csv"]["cleantestDataframePath"]
    clean_val_csv_path = hyper_params["csv"]["cleanvalDataframePath"]

    train_csv_path = os.path.join(root_dir, train_csv_path)
    test_csv_path = os.path.join(root_dir, test_csv_path)
    val_csv_path = os.path.join(root_dir, val_csv_path)

    clean_train_csv_path = os.path.join(root_dir, clean_train_csv_path)
    clean_test_csv_path = os.path.join(root_dir, clean_test_csv_path)
    clean_val_csv_path = os.path.join(root_dir, clean_val_csv_path)

    clean_dataframe(
        csv_path=train_csv_path,
        new_csv_path=clean_train_csv_path,
        punctuation=punctuation,
        stop_words=stop_words,
        number_name=number_name,
    )
    clean_dataframe(
        csv_path=test_csv_path,
        new_csv_path=clean_test_csv_path,
        punctuation=punctuation,
        stop_words=stop_words,
        number_name=number_name,
    )
    clean_dataframe(
        csv_path=val_csv_path,
        new_csv_path=clean_val_csv_path,
        punctuation=punctuation,
        stop_words=stop_words,
        number_name=number_name,
    )
    print(
        "--------------------\nTrain, test and val dataset cleaning successfull!\n--------------------\n"
    )


if __name__ == "__main__":
    main()
