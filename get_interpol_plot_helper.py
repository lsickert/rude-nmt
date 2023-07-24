"""
helper function to create the interpolation plots for the attention attributions
that can be run indendently of the main script
"""
import json
from datasets import load_from_disk
import inseq
from inseq.utils.typing import TokenWithId
from tqdm.auto import tqdm
from iwslt import preprocess
from tatoeba import DATA_FOLDER
from rude_nmt import analysis


def format_sentence(target_sent: list[TokenWithId]) -> str:
    """create a sentence from the target tokens"""
    sent = ""
    for token in target_sent:
        sent += " " + token.token[1:] if token.token.startswith("_") else token.token
    return sent


if __name__ == "__main__":

    ds = load_from_disk(DATA_FOLDER / "tatoeba_filtered")
    iwslt_en_only = preprocess.get_iwslt(
        categories=["telephony", "topical_chat", "test"], languages=["en"]
    )

    all_attr = []

    models = [
        {
            "id": "facebook/mbart-large-50-many-to-many-mmt",
            "combinations": [
                ("de_DE", "ko_KR", "tatoeba", "source", "MBart50 DE -> KO"),
                ("ko_KR", "de_DE", "tatoeba", "target", "MBart50 KO -> DE"),
                ("de_DE", "en_XX", "tatoeba", "source", "MBart50 DE -> EN"),
                ("ko_KR", "en_XX", "tatoeba", "target", "MBart50 KO -> EN"),
                ("en_XX", "de_DE", "iwslt", "en", "MBart50 EN -> DE"),
                ("en_XX", "ko_KR", "iwslt", "en", "MBart50 EN -> KO"),
            ],
        },
        {
            "id": "facebook/nllb-200-distilled-600M",
            "combinations": [
                ("deu_Latn", "kor_Hang", "tatoeba", "source", "NLLB DE -> KO"),
                ("kor_Hang", "deu_Latn", "tatoeba", "target", "NLLB KO -> DE"),
                ("deu_Latn", "eng_Latn", "tatoeba", "source", "NLLB DE -> EN"),
            ],
        },
        {
            "id": "Helsinki-NLP/opus-mt-de-en",
            "combinations": [
                (None, None, "tatoeba", "source", "OPUS DE -> EN"),
            ],
        },
        {
            "id": "Helsinki-NLP/opus-mt-ko-de",
            "combinations": [
                (None, None, "tatoeba", "target", "OPUS EN -> DE"),
            ],
        },
        {
            "id": "Helsinki-NLP/opus-mt-tc-big-en-ko",
            "combinations": [
                (None, None, "iwslt", "en", "OPUS EN -> KO"),
            ],
        },
    ]

    for model in models:
        for combination in model["combinations"]:
            inseq_model = None
            if combination[0] is None:
                inseq_model = inseq.load_model(model["id"], "attention")
            else:
                inseq_model = inseq.load_model(
                    model["id"],
                    "attention",
                    tokenizer_kwargs={
                        "src_lang": combination[0],
                        "tgt_lang": combination[1],
                    },
                )

            inputs = (
                ds[0:1600][combination[3]]
                if combination[2] == "tatoeba"
                else iwslt_en_only[combination[3]]
            )

            generation_args = (
                {
                    "forced_bos_token_id": inseq_model.tokenizer.lang_code_to_id[
                        combination[1]
                    ],
                    "max_new_tokens": 100,
                }
                if combination[0] is not None
                else {}
            )
            source_attr = []
            target_sents = []

            def batch(li, n=1):
                l = len(li)
                for ndx in range(0, l, n):
                    yield li[ndx : min(ndx + n, l)]

            with tqdm(total=len(list(batch(inputs, 8))), desc=combination[4]) as pbar:
                for texts in batch(inputs, 8):

                    out = inseq_model.attribute(
                        input_texts=texts,
                        generation_args=generation_args,
                        attribute_target=False,
                        show_progress=False,
                        batch_size=8,
                        device=inseq.utils.get_default_device(),
                    )

                    source_attr.extend(
                        [attr.source_attributions for attr in out.sequence_attributions]
                    )

                    target_sents.extend(
                        [
                            format_sentence(attr.target)
                            for attr in out.sequence_attributions
                        ]
                    )
                    pbar.update(1)

            attr_merged = analysis.format_attributions(source_attr)

            attr_output = {
                "title": combination[4],
                "merged_attributions": attr_merged,
                "target_sents": target_sents,
            }

            analysis.create_interpolation_plot_single(
                attr_merged,
                plt_title=combination[4],
                save=True,
                plt_name=f"plots/{combination[4]}.png",
            )

            all_attr.append(attr_output)

    analysis.create_interpolation_plot_multi(
        all_attr, save=True, plt_name="plots/interpolations.png"
    )

    attr_json = json.dumps(all_attr, indent=4)

    with open("interpolations.json", "w", encoding="utf-8") as f:
        f.write(attr_json)
