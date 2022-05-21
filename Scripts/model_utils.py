import fasttext
import fasttext.util
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Bidirectional, Embedding, Dense, Dropout, SimpleRNN, LSTM


def save_pickle_file(data, filename, flag=False):
    if flag:
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, "wb") as f:
            pickle.dump(data, f)


def load_pickle_file(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def load_tokenizer_word_dict(
    dataframe, tokenizer_path: str, word_dict_path: str, oov_token: str, pad_token: str
):
    try:
        tokenizer = load_pickle_file(tokenizer_path)
        word_dict = load_pickle_file(word_dict_path)
        print("Tokenizers and word dict loaded from memory!")
    except:
        tokenizer = Tokenizer(oov_token=oov_token)
        tokenizer.fit_on_texts(dataframe["Text"])
        word_dict = tokenizer.word_index
        word_dict[pad_token] = 0
        word_dict = {v: k for k, v in word_dict.items()}
        save_pickle_file(word_dict, word_dict_path)
        save_pickle_file(tokenizer, tokenizer_path, True)
    return tokenizer, word_dict


def load_embedding_matrix(
    word_dict: dict,
    vocab_size: int,
    embed_dim: int,
    embedding_matrix_path: str,
    fasttext_path: str,
):
    try:
        embedding_matrix = np.load(embedding_matrix_path + ".npy")
        print("Embedding Matrix loaded from memory!")
    except:
        ft = fasttext.load_model(fasttext_path)
        if embed_dim < 300:
            fasttext.util.reduce_model(ft, embed_dim)
        embed_dim = embed_dim
        embedding_matrix = np.zeros((vocab_size, embed_dim))
        for token, index in word_dict.items():
            if index < vocab_size:
                embed_vector = ft.get_word_vector(token)
                if embed_vector is not None:
                    embedding_matrix[index] = embed_vector
        np.save(embedding_matrix_path, embedding_matrix)
        print("Created Embedding Matrix!")
    return embedding_matrix


class Model:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        word_dict,
        embed_dim,
        sequence_length,
        embedding_matrix,
        stage_flag,
        model_flag,
        bidirectional_flag,
        model_units,
        hidden_units,
        classes,
        dropout,
        epochs,
        batch_size,
        threshold,
        checkpoint_path,
        checkpoint_monitor,
        paitence_value,
        checkpoint_mode,
        metrics_label,
        metrics_path,
        image_path,
    ):
        super(Model, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.vocab_size = len(word_dict)
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.embedding_matrix = [embedding_matrix]
        self.stage_flag = stage_flag
        self.model_flag = model_flag
        self.bidirectional_flag = bidirectional_flag
        self.model_units = model_units
        self.hidden_units = hidden_units
        self.classes = classes
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.checkpoint_path = checkpoint_path
        self.checkpoint_monitor = checkpoint_monitor
        self.paitence_value = paitence_value
        self.checkpoint_mode = checkpoint_mode
        self.metrics_label = metrics_label.split(",")
        self.metrics_path = metrics_path
        self.image_path = image_path
        self.early_stopping = EarlyStopping(
            monitor=self.checkpoint_monitor,
            patience=self.paitence_value,
            mode=self.checkpoint_mode,
        )
        self.save_best = ModelCheckpoint(
            self.checkpoint_path,
            save_best_only=True,
            monitor=self.checkpoint_monitor,
            mode=self.checkpoint_mode,
        )
        self.callbacks = [self.early_stopping, self.save_best]

    def create_model(self):
        model = Sequential()
        model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim,
                input_length=self.sequence_length,
                weights=self.embedding_matrix,
            )
        )
        model.add(Dropout(self.dropout))
        if self.model_flag == "rnn":
            if self.bidirectional_flag:
                model.add(
                    Bidirectional(
                        SimpleRNN(units=self.model_units, dropout=self.dropout)
                    )
                )
            else:
                model.add(SimpleRNN(units=self.model_units, dropout=self.dropout))
        else:
            if self.bidirectional_flag:
                model.add(
                    Bidirectional(LSTM(units=self.model_units, dropout=self.dropout))
                )
            else:
                model.add(LSTM(units=self.model_units, dropout=self.dropout))
        model.add(Dense(self.hidden_units, activation="relu"))
        model.add(Dense(self.classes, activation="sigmoid"))
        return model

    def loss_plot(self):
        plt.plot(self.history.history["loss"], label="Training data")
        plt.plot(self.history.history["val_loss"], label="Validation data")
        plt.title("Loss")
        plt.ylabel("Loss value")
        plt.xlabel("No. epoch")
        plt.legend(loc="upper left")
        plt.savefig(self.image_path)
        plt.show()

    def plot(self):
        self.loss_plot()

    def train(self):
        self.model = self.create_model()
        self.model.summary()
        self.model.compile(loss="binary_crossentropy", optimizer="adam")
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            validation_data=(self.X_val, self.y_val),
            batch_size=self.batch_size,
            verbose=1,
            callbacks=self.callbacks,
        )
        self.plot()

    def metrics(self):
        model = self.create_model()
        model.load_weights(filepath=self.checkpoint_path)
        y_pred = model.predict(self.X_test)
        upper, lower = 1, 0
        y_pred = np.where(y_pred > self.threshold, upper, lower)
        self.y_test = np.where(self.y_test > self.threshold, upper, lower)
        report = classification_report(
            self.y_test,
            y_pred,
            target_names=self.metrics_label,
            digits=3,
            zero_division=0,
        )
        if self.stage_flag == "Stage1":
            matrix = confusion_matrix(self.y_test, y_pred)
        else:
            matrix = multilabel_confusion_matrix(self.y_test, y_pred)
        with open(self.metrics_path, "w") as metrics_file:
            metrics_file.write(
                "--------------------\nClassification Report: {}\n--------------------\n\n--------------------\nConfusion Matrix{}\n--------------------\n".format(
                    report, matrix
                )
            )
