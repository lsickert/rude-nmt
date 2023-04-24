"""main module"""
import os
import sys
import argparse
from datasets import load_from_disk
import tatoeba
import iwslt
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
    default=False,
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
    ds_name = options.data

    ds_label_folder = tatoeba.DATA_FOLDER / f"{ds_name}_labelled"
    ds_trans_folder = tatoeba.DATA_FOLDER / f"{ds_name}_translated"

    if options.data == "tatoeba":
        tatoeba.preprocess.get_tatoeba(force=options.force_new)

        print("##### Preprocessing Tatoeba data #####")
        ds = tatoeba.preprocess.get_subtitle_dataset(options.force_new)

        if options.translate:
            ds = translation.translate_ds(ds, force_regen=options.force_new)

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

    elif options.data == "iwslt":

        print("##### Preprocessing IWSLT data #####")
        ds = iwslt.preprocess.get_iwslt(options.force_new)

        if options.label_data:
            for col in ds.column_names:
                if col.endswith("_de"):
                    ds = ds.map(
                        label_german.get_pos_tags,
                        batched=True,
                        load_from_cache_file=not options.force_new,
                        fn_kwargs={"col": col},
                    )
                elif col.endswith("_ko"):
                    ds = ds.map(
                        label_korean.get_pos_tags,
                        batched=True,
                        load_from_cache_file=not options.force_new,
                        fn_kwargs={"col": col},
                    )

            ds.save_to_disk(iwslt.DATA_FOLDER / f"{ds_name}_labelled")
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
