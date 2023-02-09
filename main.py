"""main module"""
from tatoeba import preprocess
from rude_nmt import label_german, label_korean

TATOEBA_TAR = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/deu-kor.tar"

if __name__ == "__main__":

    #preprocess.get_tatoeba(TATOEBA_TAR)

    subtitle_data = preprocess.get_subtitle_dataset()

    print(subtitle_data)

    subtitle_data = subtitle_data.map(label_german.annotate_formality, load_from_cache_file=False)
    subtitle_data = subtitle_data.map(label_korean.annotate_formality, load_from_cache_file=False)

    subtitle_data.to_csv("./data/subtitle_data.csv", num_proc=8)

    dat = subtitle_data.to_pandas()

    print(dat["de_formality"].value_counts())
    print(dat["ko_formality"].value_counts())
