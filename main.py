from tatoeba import preprocess

TATOEBA_TAR = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2021-08-07/deu-kor.tar"

if __name__ == "__main__":

    preprocess.get_tatoeba(TATOEBA_TAR)