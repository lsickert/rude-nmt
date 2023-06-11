"""helper script to get the datasets with contrastive attribution results"""
from datasets import load_from_disk
from rude_nmt import attribute
from iwslt import DATA_FOLDER

if __name__ == "__main__":

    ds = load_from_disk( DATA_FOLDER / "iwslt_labelled")

    ds_formal_informal_ko = attribute.perform_contrastive_attr(
        ds, "formal_de", "formal_ko", "informal_ko", "de", "ko", "input_x_gradient"
    )
    ds_informal_formal_ko = attribute.perform_contrastive_attr(
        ds, "informal_de", "informal_ko", "formal_ko", "de", "ko", "input_x_gradient"
    )

    ds_formal_informal_de = attribute.perform_contrastive_attr(
        ds, "formal_ko", "formal_de", "informal_de", "ko", "de", "input_x_gradient"
    )
    ds_informal_formal_de = attribute.perform_contrastive_attr(
        ds, "informal_ko", "informal_de", "formal_de", "ko", "de", "input_x_gradient"
    )

    ds_formal_informal_ko.save_to_disk( DATA_FOLDER / "iwslt_labelled_formal_informal_ko")
    ds_informal_formal_ko.save_to_disk( DATA_FOLDER / "iwslt_labelled_informal_formal_ko")

    ds_formal_informal_de.save_to_disk( DATA_FOLDER / "iwslt_labelled_formal_informal_de")
    ds_informal_formal_de.save_to_disk( DATA_FOLDER / "iwslt_labelled_informal_formal_de")
