import pandas as pd
import random
import tqdm
import re
import numpy as np
import os
import json

M4_model_set = {'chatGPT': 0, 'bloomz': 1, "dolly": 2, "davinci": 3, "cohere": 4, "llama2-fine-tuned": 5,  "jais-30b": 6, "GPT4": 7, "human": 8, "gemini-3.1-flash-lite": 9}

def load_M4(filefoleder, machine_text_only=False):
    data_new = {
    }
    folder = os.listdir(filefoleder)
    for entry in folder:
        full_path = os.path.join(filefoleder, entry)
        tt=entry[:-6]
        parts = tt.split('_')
        if len(parts)==3:
            keyname = f"{parts[-1]}_{parts[-2]}"
            data_new[keyname] = []
        else:
            keyname = parts[-1]
            data_new[keyname] = []
        with open(full_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                try:
                    data = json.loads(line)
                    # machine_text  model
                    dct = {}
                    dct['text'] = data["text"]
                    # reverse the label set machine label 0, human label 1
                    if data["label"] != 0:
                        dct['label'] = 0
                    else:
                        dct['label'] = 1
                    dct['src'] = data["model"]
                    # if machine_text_only then skip human text
                    if machine_text_only and dct['label'] == 1:
                        continue
                    data_new[keyname].append(dct)
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i}: {e}")
                    continue
    for key in data_new:
        data_new[key] = process_data_MGT(data_new[key])
        # only use 10% of the data
        # random.shuffle(data_new[key])
        # data_new[key] = data_new[key][:int(len(data_new[key]) * 0.01)]
    return data_new
  
           

def process_data_MGT(dataset):
    data_list=[]
    for i in range(len(dataset)):
        text,label,src=dataset[i]['text'],str(dataset[i]['label']),dataset[i]['src']
        data_list.append((text,label,src,i))
    return data_list


def load_M4_with_gemini(m4_path, gemini_train_path, n_gemini, seed=42, train_key='monolingual_train'):
    """Load M4 and append `n_gemini` gemini training samples to the train split.

    Returns the same dict shape as `load_M4`. Reuses `load_M4`'s label flip
    convention (raw label==0 -> tuple label '1' (human OOD); raw label!=0 ->
    tuple label '0' (machine ID)).

    Args:
        m4_path: folder containing the original SemEval2024-M4 SubtaskA jsonls.
        gemini_train_path: path to a jsonl of gemini-only generated samples
            (e.g. data/gemini-M4/_intermediate/gemini_train.jsonl).
        n_gemini: how many gemini samples to mix in. If <= 0, no augmentation.
        seed: RNG seed for which gemini samples are chosen.
        train_key: which key in the dict to augment. Default 'monolingual_train'.
    """
    data_new = load_M4(m4_path)
    if n_gemini is None or n_gemini <= 0:
        return data_new
    if train_key not in data_new:
        raise KeyError(f"train_key {train_key!r} not found in loaded M4; "
                       f"keys are {list(data_new.keys())}")

    gemini_records = []
    with open(gemini_train_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding gemini JSON on line {i}: {e}")
                continue
            dct = {'text': rec['text']}
            # mirror load_M4 label flip: raw 0 -> 1 (human/OOD), raw !=0 -> 0 (machine/ID)
            dct['label'] = 0 if rec.get('label', 0) != 0 else 1
            dct['src'] = rec.get('model', 'gemini-3.1-flash-lite')
            gemini_records.append(dct)

    if len(gemini_records) < n_gemini:
        print(f"[warn] requested {n_gemini} gemini samples but only "
              f"{len(gemini_records)} available; using all of them")
        chosen = gemini_records
    else:
        rng = random.Random(seed)
        chosen = rng.sample(gemini_records, n_gemini)

    base_id = len(data_new[train_key])
    for offset, dct in enumerate(chosen):
        # match the (text, label, src, id) tuple shape used by process_data_MGT
        data_new[train_key].append((dct['text'], str(dct['label']), dct['src'], base_id + offset))
    print(f"[load_M4_with_gemini] added {len(chosen)} gemini samples to "
          f"{train_key} (now {len(data_new[train_key])} total)")
    return data_new
