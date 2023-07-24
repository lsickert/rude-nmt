"""helper function to recalculate the translation metrics that can be run indendently of the main script"""
from datasets import load_from_disk
from rude_nmt import translation
from tatoeba import DATA_FOLDER

if __name__ == "__main__":

    print("#### Get German -> Korean translation metrics ####")

    ds = load_from_disk(DATA_FOLDER / "tatoeba_merged")

    scores = translation.get_translation_metrics(ds, "source", "target", "ko_nmt", "ko")

    print(f'BLEU: {scores["bleu"]}')
    print(f'CHRF: {scores["chrf"]}')
    print(f'COMET: {scores["comet"]}')

    print("#### Get Korean -> German translation metrics ####")

    scores = translation.get_translation_metrics(ds, "target", "source", "de_nmt", "de")

    print(f'BLEU: {scores["bleu"]}')
    print(f'CHRF: {scores["chrf"]}')
    print(f'COMET: {scores["comet"]}')
