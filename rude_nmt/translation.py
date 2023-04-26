"""this module contains functions to generate translations through mbart50"""
import os

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

from tqdm.auto import tqdm

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
    ds: Dataset, src_lang: str, trg_lang: str, batch_size: int = 32, force_regen: bool = False
) -> Dataset:
    """translate the given dataset using the pretrained model"""

    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt", src_lang=LANG_TAG_MAP[src_lang]
    )

    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )

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

    print(f"##### Tokenizing {src_lang} sentences #####")
    tokenized_ds = tokenize_data(ds, tokenizer, batch_size, LANG_COL_MAP[src_lang], force_regen)

    device = get_device()
    print(f"### using model on {device} ###")
    model.to(device)

    def translate_data(
        model: MBartForConditionalGeneration,
        tokenizer: MBart50TokenizerFast,
        ds: Dataset,
        batch_size: int,
        trg_lang: str,
    ) -> list:

        trans = []

        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, return_tensors="pt"
        )

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

    print(f"##### Translating {src_lang} sentences to {trg_lang} #####")
    trans = translate_data(model, tokenizer, tokenized_ds, batch_size, LANG_TAG_MAP[trg_lang])

    col_name = f"{trg_lang}_nmt"
    ds.add_column(name=col_name, column=trans)

    return ds
