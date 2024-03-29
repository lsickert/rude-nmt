"""methods to load the iwslt dataset"""
import re
from pathlib import Path
from typing import Optional
from datasets import Dataset, load_from_disk

DATA_FOLDER = Path(__file__).parent.parent.resolve() / "data"
IWSLT_FOLDER = DATA_FOLDER / "iwslt2023"


def _check_dir_exists(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)


_check_dir_exists(DATA_FOLDER)


def get_iwslt(
    force_renew: bool = False,
    categories: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
) -> Dataset:
    """gets the iwslt dataset from disk

    Args:
        force_renew (bool, optional): whether to force the regeneration of the dataset. Defaults to False.
        categories (list[str], optional): the categories to include. Defaults to ["telephony"].
        languages (list[str], optional): the languages to include. Defaults to ["en", "de", "ko"].

    Returns:
        Dataset: the dataset
    """

    if categories is None:
        categories = ["telephony"]

    if languages is None:
        languages = ["en", "de", "ko"]

    iwslt_ds_folder = DATA_FOLDER / "iwslt"

    if not iwslt_ds_folder.exists() or force_renew:

        dataset = Dataset.from_generator(
            _dataset_gen, gen_kwargs={"cats": categories, "langs": languages}
        )

        dataset = dataset.map(_get_formality_map, load_from_cache_file=not force_renew)

        dataset.save_to_disk(iwslt_ds_folder)

    else:
        dataset = load_from_disk(iwslt_ds_folder)

    old_cache = dataset.cleanup_cache_files()

    print(f"#### removed {old_cache} old cache files ####")

    return dataset


def _get_formality_map(example: dict) -> dict:
    """gets the formality map for a single example

    Args:
        example (dict): the example

    Returns:
        dict: the example with the formality map
    """

    for key in list(example.keys()):
        if "formal" in key:
            ws_tokens = example[key].split()
            form_map = [0] * len(ws_tokens)
            for i, word in enumerate(ws_tokens):
                if re.search(r"\[F\].+\[/F\]", word):
                    form_map[i] = 1
            example[f"ws_form_map_{key}"] = form_map
            example[key] = example[key].replace("[F]", "").replace("[/F]", "")
            example[f"ws_{key}"] = example[key].split()

    return example


def _dataset_gen(cats: list, langs: list) -> dict:
    """generator function to generate the iwslt dataset from the individual files

    Args:
        cats (list): the categories to include
        langs (list): the languages to include

    Yields:
        dict: the next example
    """
    for c in cats:
        files = IWSLT_FOLDER.glob(f"{c}.*")
        names = []
        handles = []
        for f in files:
            desc = f.name.split(".", 1)
            name = desc[1].replace(".", "_")
            if name[-2:] in langs:
                names.append(name)
                handles.append(open(f, "r", encoding="utf-8"))

        while True:
            line_dict = {}
            for i, h in enumerate(handles):
                line = h.readline()

                if not line:
                    break

                line_dict[names[i]] = line.strip()

            if not names[0] in line_dict:
                for h in handles:
                    h.close()
                break

            line_dict["category"] = c

            yield line_dict
