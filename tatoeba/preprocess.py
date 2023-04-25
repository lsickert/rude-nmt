"""methods to retrieve and load the tatoeba dataset"""
import io
import csv
import gzip
import shutil
import tarfile
import os
from pathlib import Path
from functools import partial
from typing import Any
import requests
from tqdm import tqdm
from datasets import Dataset, load_dataset, load_from_disk


TATOEBA_TAR = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/deu-kor.tar"
DATA_FOLDER = Path(__file__).parent.parent.resolve() / "data"


def _check_dir_exists(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)


_check_dir_exists(DATA_FOLDER)


def get_tatoeba(url: str = TATOEBA_TAR, force: bool = False) -> None:
    """download and extract the tatoeba dataset"""
    fpath = _download_file(url, force)

    if not fpath.exists() or force:
        _extract_tar(fpath)

        _extract_file("release/v2021-08-07/deu-kor/train.id.gz")
        _extract_file("release/v2021-08-07/deu-kor/train.src.gz")
        _extract_file("release/v2021-08-07/deu-kor/train.trg.gz")

    split_list = ["train", "dev", "test"]

    for split in split_list:
        out_file = DATA_FOLDER / f"deu-kor.{split}.csv"

        if not out_file.exists() or force:
            id_file = open(
                DATA_FOLDER / f"release/v2021-08-07/deu-kor/{split}.id",
                "r",
                encoding="utf-8",
            )

            num_lines = _get_file_lines(id_file)

            src_file = open(
                DATA_FOLDER / f"release/v2021-08-07/deu-kor/{split}.src",
                "r",
                encoding="utf-8",
            )
            trg_file = open(
                DATA_FOLDER / f"release/v2021-08-07/deu-kor/{split}.trg",
                "r",
                encoding="utf-8",
            )

            desc = f"generating {split} split as csv"

            with tqdm(total=num_lines, desc=desc) as pbar:
                with open(out_file, "w", encoding="utf-8", newline="") as out:
                    out_writer = csv.writer(out)
                    out_writer.writerow(["id", "source", "target"])

                    while True:
                        idx = id_file.readline().strip()
                        src = src_file.readline().strip()
                        trg = trg_file.readline().strip()

                        if not idx:
                            break

                        out_writer.writerow([idx, src, trg])
                        pbar.update()

            id_file.close()
            src_file.close()
            trg_file.close()


def get_dataset(force_renew: bool = False) -> Dataset:
    """get the processed tatoeba dataset."""
    dataset = load_dataset(
        str(DATA_FOLDER),
        name="tatoeba",
        data_files={
            "train": "deu-kor.train.csv",
            "dev": "deu-kor.dev.csv",
            "test": "deu-kor.test.csv",
        },
    )

    # some entries in the dataset include a segmentation fault
    # and consist only of the error message in one of the languages
    clean_data = dataset.filter(
        lambda ex: ex["source"] is not None and ex["target"] is not None,
        num_proc=os.cpu_count(),
        load_from_cache_file=not force_renew,
    )

    old_cache = clean_data.cleanup_cache_files()

    print(f"#### removed {old_cache} old cache files ####")

    return clean_data


def get_subtitle_dataset(force_renew: bool = False) -> Dataset:
    """get the subset with subtitle data from the train-split of the tatoeba-dataset"""

    subtitle_folder = DATA_FOLDER / "subtitles"

    if force_renew or not subtitle_folder.exists():

        if force_renew and subtitle_folder.exists():
            shutil.rmtree(subtitle_folder)

        subsets = ("OpenSubtitles-v2018", "TED2020-v1")

        dataset = get_dataset(force_renew)["train"]

        subtitle_set = dataset.filter(
            lambda ex: ex["id"].startswith(subsets),
            load_from_cache_file=not force_renew,
        )

        subtitle_set = subtitle_set.map(
            _clean_examples, num_proc=os.cpu_count(), load_from_cache_file=not force_renew
        )

        subtitle_set = subtitle_set.filter(
            lambda ex: len(ex["source"].split()) < 100 and len(ex["target"].split()) < 100,
            num_proc=os.cpu_count(),
            load_from_cache_file=not force_renew,
        )

        subtitle_set.save_to_disk(subtitle_folder)

    else:
        subtitle_set = load_from_disk(subtitle_folder)

    old_cache = subtitle_set.cleanup_cache_files()

    print(f"#### removed {old_cache} old cache files ####")

    return subtitle_set


def _clean_examples(example: dict[str, Any]) -> dict[str, Any]:
    example["source"] = example["source"].strip()
    example["target"] = example["target"].strip()

    if example["source"].startswith("- "):
        example["source"] = example["source"][2:]

    if example["source"].startswith("-"):
        example["source"] = example["source"][1:]

    if example["target"].startswith("- "):
        example["target"] = example["target"][2:]

    if example["target"].startswith("-"):
        example["target"] = example["target"][1:]

    return example


def _download_file(url: str, force_redownload: bool = False) -> Path:
    """download a file"""
    fname = url.split("/")[-1]
    fpath = DATA_FOLDER / fname

    if not fpath.exists() or force_redownload:
        with requests.get(url, stream=True, timeout=5) as res:
            if res.status_code != 200:
                res.raise_for_status()
                raise RuntimeError(f"{url} returned {res.status_code} status")

            size = int(res.headers.get("Content-Length", 0))

            res.raw.read = partial(res.raw.read, decode_content=True)

            desc = f"downloading {fname}"
            with tqdm.wrapattr(res.raw, "read", total=size, desc=desc) as raw_res:
                with open(fpath, "wb") as file:
                    shutil.copyfileobj(raw_res, file)

    return fpath


def _extract_tar(file: Path) -> None:

    tar = tarfile.open(file)
    tar.extractall()
    tar.close()


def _extract_file(file) -> str:
    """extracts a file in gzip format"""
    fname = file[:-3]
    desc = f"extracting {file}"
    compfile = gzip.open(DATA_FOLDER / file)
    size = _get_file_size(compfile)
    with tqdm.wrapattr(compfile, "read", total=size, desc=desc) as comp:
        with open(DATA_FOLDER / fname, "wb") as out:
            shutil.copyfileobj(comp, out)

    return fname


def _get_file_size(f):
    """returns the size of a file object"""
    cur = f.tell()
    f.seek(0, io.SEEK_END)
    size = f.tell()
    f.seek(cur)
    return size


def _get_file_lines(f):
    """return the number of lines in a file object"""
    count = 0
    for c, l in enumerate(f):
        count = c
    f.seek(0)
    return count + 1


if __name__ == "__main__":
    get_tatoeba(TATOEBA_TAR)
