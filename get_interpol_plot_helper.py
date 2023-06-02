"""
helper function to create the interpolation plots for the attention attributions
that can be run indendently of the main script
"""
import json
from typing import Optional
from datasets import load_from_disk
import inseq
from inseq.utils.typing import TokenWithId
import matplotlib.pyplot as plt
import numpy as np
import torch
from iwslt import preprocess
from tatoeba import DATA_FOLDER


def create_interpolation_plot_single(
    arr: np.ndarray,
    plt_name: Optional[str] = "interpolation_plot",
    plt_title: Optional[str] = None,
    save: bool = False,
) -> None:
    """create a single interpolation plot"""

    fig, ax = plt.subplots()

    pcm = ax.pcolormesh(arr, vmin=0, vmax=1, cmap="Reds")

    ax.set_xlabel("Generated Tokens")
    ax.set_ylabel("Attributed Tokens")
    if plt_title is not None:
        ax.set_title(plt_title)

    fig.colorbar(pcm, ax=ax, shrink=0.8, location="right", label="Attention Weight")

    if save:
        fig.savefig(plt_name, dpi=300, bbox_inches="tight")


def create_interpolation_plot_multi(
    plot_list: list,
    plt_name: Optional[str] = "interpolation_plot",
    save: bool = False,
) -> None:
    """create multiple interpolation plots"""

    fig, ax = plt.subplots(4, 3, figsize=(8, 8), layout="constrained")

    for col in range(3):
        for row in range(4):
            plot_id = col * 3 + row
            pcm = ax[row, col].pcolormesh(
                plot_list[plot_id]["merged_attributions"], vmin=0, vmax=1, cmap="Reds"
            )

            ax[row, col].set_xlabel("Generated Tokens")
            ax[row, col].set_ylabel("Attributed Tokens")
            ax[row, col].set_title(f"{plot_list[plot_id]['title']}")

    fig.colorbar(
        pcm, ax=ax[1:3, 2], shrink=0.4, location="right", label="Attention Weight"
    )

    # fig.tight_layout()

    if save:
        fig.savefig(plt_name)


def format_attributions(
    attr_list: list,
    scale_size: Optional[tuple] = None,
    scale_factor: Optional[float] = None,
) -> np.ndarray:
    """format the attributions for the interpolation plot"""

    if scale_size is None:
        x = np.ceil(np.mean([attr.size(0) for attr in attr_list])).astype(int)
        y = np.ceil(np.mean([attr.size(1) for attr in attr_list])).astype(int)
        scale_size = (x, y)

    attr_list = [attr.unsqueeze(0).unsqueeze(0) for attr in attr_list]
    scaled_attr = [
        torch.nn.functional.interpolate(
            attr, size=scale_size, mode="bicubic", align_corners=True
        )
        .squeeze(0)
        .squeeze(0)
        for attr in attr_list
    ]

    scaled = torch.stack(scaled_attr).mean(dim=0).numpy()

    if scale_factor is not None:
        scaled = scaled * scale_factor

    arr = np.flip(scaled, axis=0)

    return arr


def format_sentence(target_sent: list[TokenWithId]) -> str:
    """create a sentence from the target tokens"""
    sent = ""
    for token in target_sent:
        sent += " " + token.token[1:] if token.startswith("_") else token.token
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
            "id": "Helsinki-NLP/opus-mt-en-ko",
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
                        "trg_lang": combination[1],
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
                    ]
                }
                if combination[0] is not None
                else {}
            )

            out = inseq_model.attribute(
                input_texts=inputs,
                generation_args=generation_args,
                attribute_target=False,
                batch_size=4,
            )

            source_attr = [
                attr.source_attributions for attr in out.sequence_attributions
            ]
            target_sents = [
                format_sentence(attr.target) for attr in out.sequence_attributions
            ]

            attr_merged = format_attributions(source_attr)

            attr_output = {
                "title": combination[4],
                "merged_attributions": attr_merged,
                "target_sents": target_sents,
            }

            create_interpolation_plot_single(
                attr_merged,
                plt_title=combination[4],
                save=True,
                plt_name=f"plots/{combination[4]}.png",
            )

            all_attr.append(attr_output)

    create_interpolation_plot_multi(
        all_attr, save=True, plt_name="plots/interpolations.png"
    )

    attr_json = json.dumps(all_attr, indent=4)

    with open("plots/interpolations.json", "w", encoding="utf-8") as f:
        f.write(attr_json)
