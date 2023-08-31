from pathlib import Path
from typing import Union
import os

import numpy as np

import torch.backends.cuda as cuda_back
import torch.cuda as cuda
import torch.backends.mps as mps

from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset, ClassLabel

from sacrebleu.metrics import CHRF, BLEU

from . import LANG_TAG_MAP, LANG_COL_MAP

MODEL_FOLDER = Path(__file__).parent.parent.resolve() / "models"


def _check_dir_exists(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)


_check_dir_exists(MODEL_FOLDER)


def get_device() -> str:
    """returns the torch device to use for the current system"

    Returns:
        str: the device to use
    """

    if cuda.is_available() and cuda_back.is_built():
        return "cuda"
    elif mps.is_available() and mps.is_built():
        return "mps"
    else:
        return "cpu"


def fine_tune_model(
    ds: Dataset,
    model_name: str,
    src_lang: str,
    trg_lang: str,
    base_model: Union[str, Path] = "facebook/mbart-large-50-many-to-many-mmt",
    num_epochs: int = 5,
    batch_size: int = 8,
    force_regen: bool = False,
) -> Path:
    """fine tune the given model on the given dataset

    Args:

        ds (Dataset): the dataset to use
        model_name (str): the name of the model to use
        src_lang (str): the source language
        trg_lang (str): the target language
        base_model (Union[str, Path], optional): the base model to use. Defaults to "facebook/mbart-large-50-many-to-many-mmt".
        num_epochs (int, optional): the number of epochs to train for. Defaults to 3.
        batch_size (int, optional): the batch size to use. Defaults to 8.
        force_regen (bool, optional): whether to force the regeneration of the dataset. Defaults to False.

    Returns:

        Path: the path to the saved model
    """

    # for testing
    #ds = ds.select(range(1000))

    def combine_formality(example):
        example["comb_formality"] = (
            example["de_formality"] + "_" + example["ko_formality"]
        )

        return example

    #ds = ds.map(combine_formality, batched=False, num_proc=os.cpu_count())

    ds = ds.cast_column("ko_formality", ClassLabel(names=ds.unique("ko_formality")))

    split_ds = ds.train_test_split(test_size=0.8, stratify_by_column="ko_formality")

    split_ds = split_ds["train"]

    split_ds = split_ds.train_test_split(
        test_size=0.2, stratify_by_column="ko_formality"
    )

    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
        src_lang=LANG_TAG_MAP[src_lang],
        tgt_lang=LANG_TAG_MAP[trg_lang],
    )

    model = MBartForConditionalGeneration.from_pretrained(base_model)

    print(f"##### Tokenizing {src_lang} sentences #####")
    tokenized_ds = tokenize_data(
        split_ds,
        tokenizer,
        batch_size,
        LANG_COL_MAP[src_lang],
        LANG_COL_MAP[trg_lang],
        force_regen,
    )

    def compute_metrics(p):
        preds, labels = p

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = BLEU(trg_lang=trg_lang)
        chrf = CHRF()

        corpus_bleu = bleu.corpus_score(decoded_preds, [decoded_labels]).score
        corpus_chrf = chrf.corpus_score(decoded_preds, [decoded_labels]).score

        len_mismatch = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_list = pred.split(" ")
            label_list = label.split(" ")
            if len(pred_list) != len(label_list):
                len_mismatch += 1

        return {
            "len_mismatch": len_mismatch / len(preds),
            "bleu": corpus_bleu,
            "chrf": corpus_chrf,
        }

    # device = get_device()
    # print(f"### using model on {device} ###")
    # model.to(device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_FOLDER / model_name,
        label_names=["labels"],
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=1e-3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        weight_decay=0.01,
        #use_mps_device=True,
        optim="adamw_torch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(output_dir=MODEL_FOLDER / model_name)


def tokenize_data(
    ds: Dataset,
    tokenizer: MBart50TokenizerFast,
    batch_size: int,
    src_tok_col: str,
    trg_tok_col: str,
    force_regen: bool = False,
) -> Dataset:
    """tokenize the dataset using the given tokenizer

    Args:
        ds (Dataset): the dataset to tokenize
        tokenizer (MBart50TokenizerFast): the tokenizer to use
        batch_size (int): the batch size to use
        src_tok_col (str): the source column to tokenize
        trg_tok_col (str): the target column to tokenize
        force_regen (bool, optional): whether to force the regeneration of the dataset. Defaults to False.

    Returns:
        Dataset: the tokenized dataset
    """

    def tokenize_function(examples, src_tok_col: str, trg_tok_col: str):
        tokens = tokenizer(
            examples[src_tok_col],
            text_target=examples[trg_tok_col],
            truncation=True,
            max_length=512,
        )
        return tokens

    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=os.cpu_count(),
        remove_columns=ds["train"].column_names,
        fn_kwargs={"src_tok_col": src_tok_col, "trg_tok_col": trg_tok_col},
        load_from_cache_file=not force_regen,
    )
    # interestingly the batching only works correctly when the format is set here instead of inside tokenize_function
    tokenized_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return tokenized_ds
