"""main module"""
import os
import sys
import argparse
from datasets import load_from_disk
import inseq
import tatoeba
import iwslt
from rude_nmt import label_german, label_korean, translation, attribute, fine_tune

DATA = ["tatoeba", "iwslt"]
LANGUAGES = ["de", "ko"]

parser = argparse.ArgumentParser(
    prog="RudeNMT", description="Label and attribute formality in NMT"
)

parser.add_argument(
    "--data", type=str, choices=DATA, help="The dataset to use", required=True
)

parser.add_argument(
    "--src_lang",
    type=str,
    choices=LANGUAGES,
    help="The source language to use",
)

parser.add_argument(
    "--trg_lang",
    type=str,
    choices=LANGUAGES,
    help="The target language to use",
)

parser.add_argument(
    "--label_data",
    action="store_true",
    default=False,
    help="Generate and save POS tags and formality labels for the dataset",
)

parser.add_argument(
    "--force_regenerate",
    action="store_true",
    default=False,
    help="Disable the use of cached versions of the datasets",
)

parser.add_argument(
    "--force_redownload",
    action="store_true",
    default=False,
    help="Force a redownload of the tatoeba dataset",
)

parser.add_argument(
    "--translate",
    action="store_true",
    default=False,
    help="Translate the dataset",
)

parser.add_argument(
    "--fine_tune",
    action="store_true",
    default=False,
    help="Fine tune the model on the dataset",
)

parser.add_argument(
    "--model_name",
    type=str,
    help="The name of the fine tuned model",
)

parser.add_argument(
    "--attribute",
    choices=inseq.list_feature_attribution_methods(),
    help="Attribute the dataset with the following method",
)

parser.add_argument(
    "--use_ds",
    type=str,
    help="Use the given previously generated dataset for further analysis. Be aware that the data might get overwritten if you run a processing step again.",
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

    if options.use_ds:
        try:
            ds = load_from_disk(tatoeba.DATA_FOLDER / options.use_ds)
            ds_name = options.use_ds
        except FileNotFoundError:
            print("The specified dataset could not be found.")
            sys.exit(1)

    if options.src_lang and options.trg_lang:
        ds_label_folder = (
            tatoeba.DATA_FOLDER
            / f"{ds_name}_{options.src_lang}_{options.trg_lang}_labelled"
        )
        ds_attrib_folder = (
            tatoeba.DATA_FOLDER
            / f"{ds_name}_{options.src_lang}_{options.trg_lang}_attributed"
        )
    else:
        ds_label_folder = tatoeba.DATA_FOLDER / f"{ds_name}_labelled"
        ds_attrib_folder = tatoeba.DATA_FOLDER / f"{ds_name}_attributed"

    if options.data == "tatoeba":

        print("##### Preprocessing Tatoeba data #####")
        if ds is None:
            tatoeba.preprocess.get_tatoeba(force=options.force_redownload)
            ds = tatoeba.preprocess.get_subtitle_dataset(options.force_regenerate)

        if options.fine_tune:
            if not options.model_name:
                print("You must specify a model name to fine tune")
                sys.exit(1)
            fine_tune.fine_tune_model(
                ds,
                options.model_name,
                options.src_lang,
                options.trg_lang,
                force_regen=options.force_regenerate,
            )

        if options.translate:

            ds_trans_folder = (
                tatoeba.DATA_FOLDER
                / f"{ds_name}_{options.src_lang}_{options.trg_lang}_translated"
            )

            if not options.src_lang or not options.trg_lang:
                print(
                    "Source and target language must be specified in order to carry out the translation"
                )
                sys.exit(1)

            if options.src_lang == options.trg_lang:
                print("Source and target language must be different")
                sys.exit(1)

            ds = translation.translate_ds(
                ds,
                options.src_lang,
                options.trg_lang,
                force_regen=options.force_regenerate,
            )

            ds.save_to_disk(ds_trans_folder)

        if options.label_data:
            ds = label_german.annotate_ds(ds, force_regen=options.force_regenerate)
            ds = label_korean.annotate_ds(ds, force_regen=options.force_regenerate)

            ds.save_to_disk(ds_label_folder)

        if options.attribute:

            if not options.src_lang or not options.trg_lang:
                print(
                    "Source and target language must be specified in order to carry out the attribution"
                )
                sys.exit(1)

            if options.src_lang == options.trg_lang:
                print("Source and target language must be different")
                sys.exit(1)

            ds = attribute.attribute_ds(
                ds,
                options.src_lang,
                options.trg_lang,
                options.attribute,
                force_regen=options.force_regenerate,
            )

            ds.save_to_disk(ds_attrib_folder)

    elif options.data == "iwslt":

        if ds is None:
            print("##### Preprocessing IWSLT data #####")
            ds = iwslt.preprocess.get_iwslt(options.force_regenerate)

        if options.label_data:
            for col in ds.column_names:
                if col.endswith("_de") and not col.startswith("ws_"):
                    ds = ds.map(
                        label_german.get_pos_tags,
                        batched=True,
                        load_from_cache_file=not options.force_regenerate,
                        fn_kwargs={"col": col},
                    )
                elif col.endswith("_ko") and not col.startswith("ws_"):
                    ds = ds.map(
                        label_korean.get_pos_tags,
                        batched=True,
                        load_from_cache_file=not options.force_regenerate,
                        fn_kwargs={"col": col},
                    )

            ds.save_to_disk(ds_label_folder)

    if options.save_csv:
        ds.to_csv(f"./data/{ds_name}_data.csv", num_proc=os.cpu_count())
