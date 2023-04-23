"""main module"""
import os
import sys
import argparse
from datasets import load_from_disk
from tatoeba import preprocess, DATA_FOLDER
from rude_nmt import label_german, label_korean, translation

DATA = ["tatoeba", "iwslt"]

parser = argparse.ArgumentParser(
    prog="RudeNMT", description="Label and attribute formality in NMT"
)

parser.add_argument(
    "--data", type=str, choices=DATA, help="The dataset to use", required=True
)

parser.add_argument(
    "--label_data",
    action="store_true",
    default=True,
    help="Generate and save POS tags and formality labels for the dataset",
)

parser.add_argument(
    "--force_new",
    action="store_true",
    default=False,
    help="Disable the use of cached versions of the datasets",
)

parser.add_argument(
    "--translate",
    action="store_true",
    default=False,
    help="Translate the dataset",
)

parser.add_argument(
    "--save_csv",
    action="store_true",
    default=False,
    help="Save the dataset as a csv file",
)

if __name__ == "__main__":

    options = parser.parse_args()

    ds = None
    ds_name = "unknown"

    ds_label_folder = DATA_FOLDER / f"{ds_name}_labelled"
    ds_trans_folder = DATA_FOLDER / f"{ds_name}_translated"

    if options.data == "tatoeba":
        ds_name = "tatoeba"
        preprocess.get_tatoeba(force=options.force_new)

        print("##### Preprocessing Tatoeba data #####")
        ds = preprocess.get_subtitle_dataset(options.force_new)

    if options.translate:
        ds = translation.translate(ds, options.force_new)

        ds.save_to_disk(ds_trans_folder)

    if options.label_data:
        ds = label_german.annotate_ds(ds, options.force_new)
        ds = label_korean.annotate_ds(ds, options.force_new)

        ds.save_to_disk(ds_label_folder)
    else:
        try:
            ds = load_from_disk(ds_label_folder)
        except FileNotFoundError:
            print(
                "No labelled dataset found. Run with --label_data to generate it at least once."
            )
            sys.exit(1)

    if options.save_csv:
        ds.to_csv(f"./data/{ds_name}_data.csv", num_proc=os.cpu_count())
