import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    traintest = pd.concat([train, test], ignore_index = True)

    currency_list = traintest["CURRENCY"].unique()
    train_local = extract(train, currency_list)
    test_local = extract(test, currency_list)

    train_local, test_local = convert_usd(train_local, test_local, train)

    train_local = normalize(train_local, train)
    test_local = normalize(test_local, train)

    train_local["LOAN_AMOUNT_DESC_LOG_Z_isna"] = train_local["LOAN_AMOUNT_DESC_LOG_Z"].isna().astype(int)
    train_local["LOAN_AMOUNT_DESC_LOG_Z"].fillna(0, inplace = True)
    test_local["LOAN_AMOUNT_DESC_LOG_Z_isna"] = test_local["LOAN_AMOUNT_DESC_LOG_Z"].isna().astype(int)
    test_local["LOAN_AMOUNT_DESC_LOG_Z"].fillna(0, inplace = True)

    train_local.to_csv("data/loan_amount_local_train.csv", index = False, header = True)
    test_local.to_csv("data/loan_amount_local_test.csv", index = False, header = True)

def extract(data, currency_list):
    data.loc[lambda df: df["DESCRIPTION_TRANSLATED"].isna(), "DESCRIPTION_TRANSLATED"] = data.loc[lambda df: df["DESCRIPTION_TRANSLATED"].isna(), "DESCRIPTION"]
    sentences = data.set_index(["LOAN_ID", "CURRENCY"])["DESCRIPTION_TRANSLATED"].str.split(".").explode()
    local = []
    for currency in tqdm(currency_list):
        local.append(
            sentences.
            loc[lambda s: s.str.contains(currency)].
            str.replace(",", "", regex = False).
            str.extractall("([0-9]+)")[0].rename("LOAN_AMOUNT_LOCAL").
            astype(float).
            groupby(["LOAN_ID", "CURRENCY"]).max().
            reset_index().
            loc[lambda df: df["CURRENCY"] == currency, :].copy()
        )
    local = pd.concat(local)
    local = data[["LOAN_ID", "CURRENCY"]].merge(local, how = "left", on = ["LOAN_ID", "CURRENCY"])

    return local

def convert_usd(train_local, test_local, train):
    rate = (
        train_local.
        merge(train[["LOAN_ID", "LOAN_AMOUNT"]], how = "left", on = "LOAN_ID").
        assign(rate = lambda df: df["LOAN_AMOUNT_LOCAL"] / df["LOAN_AMOUNT"]).
        groupby(["CURRENCY"])["rate"].median()
    )
    train_local = (
        train_local.
        merge(rate, how = "left", on = "CURRENCY").
        assign(LOAN_AMOUNT_DESC = lambda df: df["LOAN_AMOUNT_LOCAL"] / df["rate"])
        [["LOAN_ID", "LOAN_AMOUNT_DESC"]].copy()
    )
    test_local = (
        test_local.
        merge(rate, how = "left", on = "CURRENCY").
        assign(LOAN_AMOUNT_DESC = lambda df: df["LOAN_AMOUNT_LOCAL"] / df["rate"])
        [["LOAN_ID", "LOAN_AMOUNT_DESC"]].copy()
    )

    return train_local, test_local

def normalize(data, train):
    mean = np.mean(np.log(train["LOAN_AMOUNT"]))
    std = np.std(np.log(train["LOAN_AMOUNT"]))
    data["LOAN_AMOUNT_DESC_LOG_Z"] = (np.log(data["LOAN_AMOUNT_DESC"] + 1) - mean) / std
    data = data[["LOAN_ID", "LOAN_AMOUNT_DESC_LOG_Z"]].copy()

    return data

if __name__ == "__main__":
    main()
