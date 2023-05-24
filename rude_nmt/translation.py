"""this module contains functions to generate translations through mbart50"""
import os
from statistics import fmean
import torch

# import torch.backends.mps as mps
import torch.backends.cuda as cuda_back
import torch.cuda as cuda
from torch.utils.data import DataLoader

from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

from comet import download_model, load_from_checkpoint

from tqdm.auto import tqdm

from sacrebleu.metrics import CHRF, BLEU

LANG_TAG_MAP = {
    "de": "de_DE",
    "ko": "ko_KR",
}

LANG_COL_MAP = {
    "de": "source",
    "ko": "target",
}


def get_device() -> str:
    """returns the torch device to use for the current system"""

    if cuda.is_available() and cuda_back.is_built():
        return "cuda"
    # elif mps.is_available() and mps.is_built():
    # return "cpu"
    else:
        return "cpu"


def translate_ds(
    ds: Dataset,
    src_lang: str,
    trg_lang: str,
    batch_size: int = 32,
    force_regen: bool = False,
    add_metrics: bool = True,
    add_neural_metrics: bool = True,
) -> Dataset:
    """translate the given dataset using the pretrained model"""

    # for testing
    # ds = ds.select(range(64))

    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
        src_lang=LANG_TAG_MAP[src_lang],
        tgt_lang=LANG_TAG_MAP[trg_lang],
    )

    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )

    print(f"##### Tokenizing {src_lang} sentences #####")
    tokenized_ds = tokenize_data(
        ds, tokenizer, batch_size, LANG_COL_MAP[src_lang], force_regen
    )

    device = get_device()
    print(f"### using model on {device} ###")
    model.to(device)

    print(f"##### Translating {src_lang} sentences to {trg_lang} #####")
    trans = translate_data(
        model, tokenizer, tokenized_ds, batch_size, LANG_TAG_MAP[trg_lang], device
    )

    col_name = f"{trg_lang}_nmt"
    ds = ds.add_column(name=col_name, column=trans)

    if add_metrics:
        print("##### adding statistical metrics #####")
        chrf = CHRF()
        bleu = BLEU(effective_order=True, trg_lang=trg_lang)
        ds = ds.map(
            get_stat_metrics,
            load_from_cache_file=not force_regen,
            fn_kwargs={
                "hyp_col": col_name,
                "ref_col": LANG_COL_MAP[trg_lang],
                "chrf_func": chrf,
                "bleu_func": bleu,
            },
        )
        corpus_bleu = bleu.corpus_score(
            ds[col_name], [ds[LANG_COL_MAP[trg_lang]]]
        ).score
        corpus_chrf = chrf.corpus_score(
            ds[col_name], [ds[LANG_COL_MAP[trg_lang]]]
        ).score
        print(f"#### CHRF Score: {corpus_bleu}")
        print(f"#### BLEU Score: {corpus_chrf}")

    if add_neural_metrics:
        print("##### adding neural metrics #####")
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_model_path)

        comet_ds = ds.map(
            get_comet_format,
            num_proc=os.cpu_count(),
            load_from_cache_file=not force_regen,
            fn_kwargs={
                "src_col": LANG_COL_MAP[src_lang],
                "ref_col": LANG_COL_MAP[trg_lang],
                "hyp_col": col_name,
            },
            remove_columns=ds.column_names,
        )
        comet_output = comet_model.predict(
            comet_ds.to_list(),
            batch_size=batch_size,
            gpus=1 if device == "cuda" else 0,
            accelerator=device,
        )
        ds = ds.add_column(
            name=f"comet_{trg_lang}", column=round(comet_output["scores"], 3)
        )
        print(f"#### COMET Score: {round(comet_output['system_score'],3)}")

    return ds


def tokenize_data(
    ds: Dataset,
    tokenizer: MBart50TokenizerFast,
    batch_size: int,
    tok_col: str,
    force_regen: bool = False,
) -> Dataset:
    """tokenize the dataset using the given tokenizer"""

    def tokenize_function(examples, tok_col: str):
        tokens = tokenizer(
            examples[tok_col],
            truncation=True,
            max_length=512,
        )
        return tokens

    tokenized_ds = ds.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=os.cpu_count(),
        remove_columns=ds.column_names,
        fn_kwargs={"tok_col": tok_col},
        load_from_cache_file=not force_regen,
    )
    # interestingly the batching only works correctly when the format is set here instead of inside tokenize_function
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized_ds


def translate_data(
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    ds: Dataset,
    batch_size: int,
    trg_lang: str,
    device: str = "cpu",
) -> list:
    """translate the given dataset using the given model and tokenizer"""

    trans = []

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

    loaded_ds = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)

    progress_bar = tqdm(range(len(loaded_ds)))

    model.eval()
    model.zero_grad()

    for batch in loaded_ds:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}

            batch_len = batch["input_ids"].size(1)

            max_length = 2 * batch_len

            outputs = model.generate(
                **batch,
                forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang],
                max_new_tokens=max_length,
            )

        batch_trans = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        trans.extend(batch_trans)

        progress_bar.update(1)

    progress_bar.close()
    return trans


def get_stat_metrics(
    example, hyp_col: str, ref_col: str, trg_lang: str, chrf_func: CHRF, bleu_func: BLEU
):
    """get the BLEU and CHRF scores for the given example"""

    chrf_score = chrf_func.sentence_score(example[hyp_col], [example[ref_col]])
    bleu_score = bleu_func.sentence_score(example[hyp_col], [example[ref_col]])

    example[f"chrf_{trg_lang}"] = round(chrf_score.score, 3)
    example[f"bleu_{trg_lang}"] = round(bleu_score.score, 3)

    return example


def get_comet_format(example, src_col: str, hyp_col: str, ref_col: str):
    """format the example for use with COMET"""
    com_sample = {
        "src": example[src_col],
        "mt": example[hyp_col],
        "ref": example[ref_col],
    }

    return com_sample


def get_translation_metrics(
    ds, src_col: str, ref_col: str, hyp_col: str, trg_lang: str
) -> dict:
    """helper function to calculate the BLEU, chrF and COMET score for a subset of the full dataset"""
    bleu = BLEU(trg_lang=trg_lang)
    chrf = CHRF()

    corpus_bleu = bleu.corpus_score(ds[hyp_col], [ds[ref_col]]).score
    corpus_chrf = chrf.corpus_score(ds[hyp_col], [ds[ref_col]]).score

    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)

    comet_ds = ds.map(
        get_comet_format,
        num_proc=os.cpu_count(),
        fn_kwargs={
            "src_col": src_col,
            "ref_col": ref_col,
            "hyp_col": hyp_col,
        },
        remove_columns=ds.column_names,
    )

    comet_output = comet_model.predict(
        comet_ds.to_list(),
        batch_size=32,
        gpus=1 if get_device() == "cuda" else 0,
        accelerator=get_device(),
    )

    return {
        "bleu": corpus_bleu,
        "chrf": corpus_chrf,
        "comet": round(comet_output["system_score"], 3)
    }
