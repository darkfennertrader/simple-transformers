import os
import json
import pandas as pd

pd.set_option("max_colwidth", 120)
pd.set_option("display.max_columns", 1)
pd.set_option("display.width", 1980)


def data_prep(directory, filemane):
    _dir = directory
    filename = filemane
    with open(_dir + filename, "r", encoding="utf-8") as f:
        raw_data = json.loads(f.read())

    dialogues = []
    prev_length = 0
    for i in range(len(raw_data["utterances"])):
        if len(raw_data["utterances"][i]["history"]) > prev_length:
            prev_length = len(raw_data["utterances"][i]["history"])
        else:
            dialogues.append(
                " <|endoftext|> ".join(raw_data["utterances"][i - 1]["history"]) + " \n"
            )
            prev_length = 0

    dialogues.append(" <|endoftext|> ".join(raw_data["utterances"][i]["history"]))

    dataset = pd.DataFrame(dialogues, columns=["dialogues"])
    # print()
    # print(dataset.head())
    # print()

    with open(_dir + filename.split(".")[0] + "_processed.txt", "w") as f:
        for i, dialog in enumerate(dialogues):
            f.write(dialog)

    return dataset
