import tarfile
import gzip
import shutil
import io
from pathlib import Path
from functools import partial
import csv
import requests
from tqdm import tqdm


TATOEBA_TAR = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/deu-kor.tar"
DATA_FOLDER = Path(__file__).parent.parent.resolve() / "data"


def _check_dir_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True)


_check_dir_exists(DATA_FOLDER)


def download_file(url: str, force_redownload: bool = False) -> Path:
    """download a file"""
    fname = url.split("/")[-1]
    fpath = DATA_FOLDER / fname

    if not fpath.exists() or force_redownload:
        with requests.get(url, stream=True) as res:
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


def get_tatoeba(url: str, force: bool = False):
    """download and extract the tatoeba dataset"""
    fpath = download_file(url, force)

    _extract_tar(fpath)

    _extract_file("release/v2021-08-07/deu-kor/train.id.gz")
    _extract_file("release/v2021-08-07/deu-kor/train.src.gz")
    _extract_file("release/v2021-08-07/deu-kor/train.trg.gz")

    split_list = ["train", "dev", "test"]

    for split in split_list:
        out_file = DATA_FOLDER / f"deu-kor.{split}.csv"

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


def _extract_tar(file: Path):

    tar = tarfile.open(file)
    tar.extractall()
    tar.close()


def _extract_file(file) -> str:
    """extracts a file from dbpedia in gzip format"""
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
