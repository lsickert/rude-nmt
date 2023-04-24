"""methods to load the iwslt dataset"""
from pathlib import Path
from datasets import Dataset, load_from_disk

DATA_FOLDER = Path(__file__).parent.parent.resolve() / "data"
IWSLT_FOLDER = DATA_FOLDER / "iwslt2023"


def _check_dir_exists(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)


_check_dir_exists(DATA_FOLDER)


def get_iwslt(force_renew: bool = False) -> Dataset:
    """load the iwslt dataset"""

    cats = ["telephony"]

    iwslt_ds_folder = DATA_FOLDER / "iwslt"

    if not iwslt_ds_folder.exists() or force_renew:

        dataset = Dataset.from_generator(_dataset_gen, gen_kwargs={"cat": cats})

        dataset.save_to_disk(iwslt_ds_folder)

    else:
        dataset = load_from_disk(iwslt_ds_folder)

    old_cache = dataset.cleanup_cache_files()

    print(f"#### removed {old_cache} old cache files ####")

    return dataset


def _dataset_gen(cat: list) -> dict:
    """generator function to generate the iwslt dataset from the individual files"""
    for c in cat:
        files = IWSLT_FOLDER.glob(f"{c}.*")
        names = []
        handles = []
        for f in files:
            desc = f.name.split(".", 1)
            name = desc[1].replace(".", "_")
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
