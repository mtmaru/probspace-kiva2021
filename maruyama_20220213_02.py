import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
import pickle
import pathlib

def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename = "log/{:s}.log".format(pathlib.PurePath(__file__).stem))
    logger.addHandler(handler)

    set_seed(0)

    # データセット作成
    category_tokenizer = CategoryTokenizer().fit(pd.read_csv("data/train.csv"))
    dataset = Dataset("data/train.csv", category_tokenizer)
    dataset_test = Dataset("data/test.csv", category_tokenizer)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

    # エポック終了時点のモデルを保存するコールバック関数
    def save_snapshot(epoch, estimator, locals):
        with open("model/{:s}_epoch{:02d}.pickle".format(pathlib.PurePath(__file__).stem, epoch + 1), mode = "wb") as fp:
            pickle.dump(estimator, fp)
        return False

    # 目的変数を元に戻す関数
    def denormalize(y):
        return np.exp(y * dataset.y_log_std + dataset.y_log_mean)

    # エポック終了後に検証データでMAEを計算するコールバック関数
    def print_mae_epoch(epoch, estimator, locals):
        val_y = denormalize(dataset.y_log_z[dataset_val.indices].cpu().detach().numpy())
        val_y_pred = denormalize(estimator.predict(dataset_val))
        val_mae = np.mean(np.abs(val_y - val_y_pred))
        message = "Epoch: {:4d}, MAE: {:.4f}".format(epoch + 1, val_mae)
        logger.info(message)
        tqdm.write(message)
        return False

    # ステップ終了後に学習データのミニバッチでMAEを計算するコールバック関数
    history = {}
    def print_mae_step(epoch, step, estimator, locals):
        if epoch not in history:
            history[epoch] = []
        batch_y = denormalize(locals["batch"][-1].cpu().detach().numpy())
        batch_y_pred = denormalize(locals["batch_y_pred"].cpu().detach().numpy())
        batch_mae = np.mean(np.abs(batch_y - batch_y_pred))
        history[epoch].append(batch_mae)
        if (step + 1) % 10 == 0:
            message = "Epoch: {:4d}, Step: {:4d}, MAE: {:9.4f} ± {:9.4f}".format(
                epoch + 1,
                step + 1,
                np.mean(np.array(history[epoch])),
                np.std(np.array(history[epoch]))
            )
            logger.info(message)
            tqdm.write(message)
        return False

    # 学習
    estimator = Estimator(
        epochs = 10,
        batch_size = 32,
        num_embeddings = category_tokenizer.dictionary.shape[0] + 1,
        embedding_dim = 8
    )
    estimator = estimator.fit(dataset_train, [save_snapshot, print_mae_epoch], [print_mae_step])

    # 予測
    for epoch in range(7, 10):
        with open("model/{:s}_epoch{:02d}.pickle".format(pathlib.PurePath(__file__).stem, epoch + 1), mode = "rb") as fp:
            estimator = pickle.load(fp)
        test_y_pred = estimator.predict(dataset_test)
        submit = pd.DataFrame({
            "LOAN_ID": dataset_test.data["LOAN_ID"],
            "LOAN_AMOUNT": denormalize(test_y_pred)
        })
        submit.to_csv("submit/{:s}_epoch{:02d}.csv".format(pathlib.PurePath(__file__).stem, epoch + 1), index = False, header = True)

def set_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, category_tokenizer):
        self.csv_path = csv_path
        self.category_tokenizer = category_tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__preprocess()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return (
            self.category_variables[index, :].to(self.device),
            self.data.loc[index, "DESCRIPTION_TRANSLATED"],
            self.data.loc[index, "LOAN_USE"],
            self.tags[index, :].to(self.device),
            self.y_log_z[index].to(self.device)
        )

    def __preprocess(self):
        # 画像以外のデータはあらかじめすべて読み込んでおく
        self.data = pd.read_csv(self.csv_path)

        # 目的変数を標準化
        self.y_log_z, self.y_log_mean, self.y_log_std = self.__preprocess_normalize_objective_variable(self.data)

        # カテゴリー変数の各水準をIDに変換
        self.category_variables = self.category_tokenizer.transform(self.data)

        # 説明の欠損を補完
        self.data.loc[lambda df: df["DESCRIPTION_TRANSLATED"].isna(), "DESCRIPTION_TRANSLATED"] = self.data.loc[lambda df: df["DESCRIPTION_TRANSLATED"].isna(), "DESCRIPTION"]

        # タグをフラグ化
        self.tags = self.__preprocess_encode_tags(self.data)

    def __preprocess_normalize_objective_variable(self, data):
        # 学習データ
        if "LOAN_AMOUNT" in data.columns:
            y_log = np.log(data["LOAN_AMOUNT"].values)
            y_log_mean = np.mean(y_log)
            y_log_std = np.std(y_log)
            y_log_z = (y_log - y_log_mean) / y_log_std
            y_log_z = torch.from_numpy(y_log_z).float()
        # テストデータ
        else:
            y_log_mean = 0
            y_log_std = 1
            y_log_z = np.zeros(data.shape[0])
            y_log_z = torch.from_numpy(y_log_z).float()

        return y_log_z, y_log_mean, y_log_std

    def __preprocess_encode_tags(self, data):
        # タグ一覧 (全30種＋欠損)
        master = [
            "animals",
            "biz durable asset",
            "eco-friendly",
            "elderly",
            "fabrics",
            "female education",
            "first loan",
            "health and sanitation",
            "job creator",
            "na",
            "orphan",
            "parent",
            "refugee",
            "repair renew replace",
            "repeat borrower",
            "schooling",
            "single",
            "single parent",
            "supporting family",
            "sustainable ag",
            "technology",
            "trees",
            "unique",
            "us black-owned business",
            "us immigrant",
            "user_favorite",
            "vegan",
            "volunteer_like",
            "volunteer_pick",
            "widowed",
            "woman-owned business"
        ]

        # フラグ化
        tags = (
            data["TAGS"].
            fillna("NA").  # タグがない場合は、"NA" タグを付ける
            str.split(",", expand = True).
            stack().
            str.strip().
            str.replace("#", "", regex = False).
            str.lower().
            rename_axis(["index", "order"]).
            rename("tag").
            reset_index().
            drop(columns = ["order"]).
            drop_duplicates().
            assign(one = 1).
            pivot(index = "index", columns = "tag", values = "one").
            fillna(0.0)
        )

        # データにないタグの列を追加
        for tag in master:
            if tag not in tags.columns:
                tags[tag] = 0.0
        tags = tags[master]

        tags = torch.from_numpy(tags.values).float()

        return tags

class CategoryTokenizer():
    def __init__(self):
        pass

    def fit(self, x):
        dictionary = self.__select_category_variables(x)
        dictionary = dictionary.melt()
        dictionary = dictionary.drop_duplicates(ignore_index = True).reset_index().rename(columns = {"index": "id"})
        dictionary["id"] = dictionary["id"] + 1
        self.dictionary = dictionary

        return self

    def transform(self, x):
        x = self.__select_category_variables(x)
        x = x.melt(ignore_index = False).reset_index()
        x = x.merge(self.dictionary, how = "left", on = ["variable", "value"])
        x["id"] = x["id"].fillna(0)  # 辞書にないカテゴリーに0番を割り当てる
        x = x.pivot(index = "index", columns = "variable", values = "id")
        x = torch.from_numpy(x.values).int()

        return x

    def __select_category_variables(self, x):
        x = x.copy()
        x["SECTOR_NAME > ACTIVITY_NAME"] = x["SECTOR_NAME"] + " > " + x["ACTIVITY_NAME"]
        x["COUNTRY_CODE > COUNTRY_NAME"] = x["COUNTRY_CODE"] + " > " + x["COUNTRY_NAME"]
        x["COUNTRY_CODE > COUNTRY_NAME > TOWN_NAME"] = x["COUNTRY_CODE"] + " > " + x["COUNTRY_NAME"] + " > " + x["TOWN_NAME"]
        x["CURRENCY_EXCHANGE_COVERAGE_RATE"] = x["CURRENCY_EXCHANGE_COVERAGE_RATE"].astype(str)
        x["CURRENCY_POLICY > CURRENCY_EXCHANGE_COVERAGE_RATE"] = x["CURRENCY_POLICY"] + " > " + x["CURRENCY_EXCHANGE_COVERAGE_RATE"]
        x = x[[
            "ORIGINAL_LANGUAGE",
            "SECTOR_NAME",
            "SECTOR_NAME > ACTIVITY_NAME",
            "COUNTRY_CODE > COUNTRY_NAME",
            "COUNTRY_CODE > COUNTRY_NAME > TOWN_NAME",
            "CURRENCY_POLICY > CURRENCY_EXCHANGE_COVERAGE_RATE",
            "CURRENCY",
            "REPAYMENT_INTERVAL",
            "DISTRIBUTION_MODEL"
        ]].copy()

        return x

class Estimator:
    def __init__(self, epochs, batch_size, num_embeddings, embedding_dim):
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def __getstate__(self):
        state = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "model": self.model.state_dict()
        }

        return state

    def __setstate__(self, state):
        self.epochs = state["epochs"]
        self.batch_size = state["batch_size"]
        self.num_embeddings = state["num_embeddings"]
        self.embedding_dim = state["embedding_dim"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(self.num_embeddings, self.embedding_dim).to(self.device)
        self.model.load_state_dict(state["model"])

    def fit(self, dataset, callbacks_epoch = [], callbacks_step = []):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        self.model = self.__init_model()
        criterion = nn.MSELoss()
        optimizer = self.__init_optimizer(self.model)
        for epoch in tqdm(range(self.epochs), desc = "epoch (fit)"):
            self.model.train()
            for step, batch in enumerate(tqdm(dataloader, desc = "step (fit)")):
                optimizer.zero_grad()
                batch_y_pred = self.model(batch[:-1])
                loss = criterion(batch_y_pred, batch[-1])
                loss.backward()
                optimizer.step()
                locals_snapshot = locals()
                if any([callback(epoch, step, self, locals_snapshot) for callback in callbacks_step]):
                    break
            locals_snapshot = locals()
            if any([callback(epoch, self, locals_snapshot) for callback in callbacks_epoch]):
                break

        return self

    def predict(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = False)
        self.model.eval()
        y_pred = []
        for step, batch in enumerate(tqdm(dataloader, desc = "step (predict)")):
            with torch.no_grad():
                batch_y_pred = self.model(batch[:-1])
                y_pred.append(batch_y_pred.cpu().detach().numpy())
        y_pred = np.concatenate(y_pred)

        return y_pred

    def __init_model(self):
        model = Model(self.num_embeddings, self.embedding_dim).to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.category_encoder.parameters():
            param.requires_grad = True
        for param in model.text_encoder.encoder.layer[-1].parameters():  # BERTの最終層だけfine-tuningする
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

        return model

    def __init_optimizer(self, model):
        optimizer = torch.optim.AdamW([
            {"params": model.category_encoder.parameters(), "lr": 0.0010},
            {"params": model.text_encoder.encoder.layer[-1].parameters(), "lr": 0.0005},
            {"params": model.fc.parameters(), "lr": 0.0010}
        ], eps = 1e-6, weight_decay = 1e-4)

        return optimizer

class Model(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_category_variables = 9
        self.num_tags = 31

        self.category_encoder = nn.Embedding(num_embeddings = num_embeddings, embedding_dim = embedding_dim)

        self.text_tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
        self.text_encoder = transformers.RobertaModel.from_pretrained("roberta-base")

        self.fc = nn.Sequential(
            nn.Linear(in_features = self.num_category_variables * embedding_dim + 768 + 768 + self.num_tags, out_features = 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features = 128, out_features = 1)
        )

    def forward(self, batch):
        # (batch_size, num_category_variables)
        category_variables = batch[0]
        # (batch_size, )
        text_dsc = batch[1]
        # (batch_size, )
        text_use = batch[2]
        # (batch_size, num_tags)
        tags = batch[3]

        # (batch_size, num_category_variables, embedding_dim)
        category_variables = self.category_encoder(category_variables)
        # (batch_size, num_category_variables * embedding_dim)
        category_variables = category_variables.view(category_variables.shape[0], category_variables.shape[1] * category_variables.shape[2])

        # (batch_size, 768)
        text_dsc = self.__foward_text(text_dsc)
        # (batch_size, 768)
        text_use = self.__foward_text(text_use)

        # (batch_size, num_category_variables * embedding_dim + 768 + 768 + num_tags)
        x = torch.cat([category_variables, text_dsc, text_use, tags], dim = 1)
        # (batch_size, 1)
        x = self.fc(x)
        # (batch_size)
        x = torch.flatten(x)

        return x

    def __foward_text(self, text_variable):
        # text_variable: (batch_size, )

        # (batch_size, num_words, 768), (batch_size, num_words, 768)
        tokenized = self.text_tokenizer(text_variable, padding = True, truncation = True, max_length = 512, return_tensors = "pt")
        tokenized["input_ids"] = tokenized["input_ids"].to(self.device)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(self.device)
        # (batch_size, num_words, 768)
        text_variable = self.text_encoder(**tokenized)["last_hidden_state"]
        # (batch_size, 768)
        text_variable = text_variable[:, 0, :]  # cls

        # (batch_size, 768)
        return text_variable

if __name__ == "__main__":
    main()
