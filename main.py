"""main module"""
import os
from tatoeba import preprocess, DATA_FOLDER
from rude_nmt import label_german, label_korean

TATOEBA_TAR = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/deu-kor.tar"

if __name__ == "__main__":

    preprocess.get_tatoeba(TATOEBA_TAR)

    subtitle_data = preprocess.get_subtitle_dataset()

    subtitle_data = subtitle_data.map(label_german.get_pos_tags, load_from_cache_file=False, batched=True)
    subtitle_data = subtitle_data.map(label_korean.get_pos_tags, load_from_cache_file=False, batched=True)
    subtitle_data = subtitle_data.map(label_german.annotate_formality, load_from_cache_file=False, num_proc=os.cpu_count())
    subtitle_data = subtitle_data.map(label_korean.annotate_formality, load_from_cache_file=False, num_proc=os.cpu_count())

    subtitle_data.to_csv("./data/subtitle_data.csv", num_proc=os.cpu_count())

    subtitle_folder = DATA_FOLDER / "subtitles_labelled"

    subtitle_data.save_to_disk(subtitle_folder)

