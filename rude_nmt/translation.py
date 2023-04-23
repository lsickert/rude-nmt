import os
from pathlib import Path

# import torch.backends.mps as mps
import torch.backends.cuda as cuda_back
import torch.cuda as cuda
from torch.utils.data import DataLoader

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset

from tqdm.auto import tqdm


def get_device() -> str:
    """returns the torch device to use for the current system"""

    if cuda.is_available() and cuda_back.is_built():
        return "cuda"
    # elif mps.is_available() and mps.is_built():
    # return "cpu"
    else:
        return "cpu"


def translate_ds(
    ds: Dataset, batch_size: int = 32, force_regen: bool = False
) -> Dataset:
    """translate the given dataset using the pretrained model"""

    tokenizer_de = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt", src_lang="de_DE"
    )
    tokenizer_ko = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt", src_lang="ko_KR"
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
                padding=True,
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

    print("##### Tokenizing German sentences #####")
    tokenized_ds_de = tokenize_data(ds, tokenizer_de, batch_size, "source", force_regen)
    print("##### Tokenizing Korean sentences #####")
    tokenized_ds_ko = tokenize_data(ds, tokenizer_ko, batch_size, "target", force_regen)

    device = get_device()
    print(f"### evaluating model on {device} ###")
    model.to(device)

    model.eval()
    model.zero_grad()

    def translate_data(
        model: MBartForConditionalGeneration,
        tokenizer: MBart50TokenizerFast,
        ds: Dataset,
        batch_size: int,
        trg_lang: str,
    ) -> list:

        trans = []

        loaded_ds = DataLoader(ds, batch_size=batch_size)

        progress_bar = tqdm(range(len(loaded_ds)))

        for batch in loaded_ds:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model.generate(
                **batch, forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang]
            )

            batch_trans = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            trans.extend(batch_trans)

            progress_bar.update(1)

        progress_bar.close()
        return trans

    print("##### Translating German sentences #####")
    ko_trans = translate_data(model, tokenizer_de, tokenized_ds_de, batch_size, "ko_KR")
    print("##### Translating Korean sentences #####")
    de_trans = translate_data(model, tokenizer_ko, tokenized_ds_ko, batch_size, "de_DE")

    ds.add_column(name="ko_nmt", column=ko_trans)
    ds.add_column(name="de_nmt", column=de_trans)

    return ds
