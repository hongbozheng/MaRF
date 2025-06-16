from .registry import register_model
from transformers import BertConfig, BertModel


@register_model(name="bert")
def build_model(cfg) -> BertModel:
    bert_cfg = BertConfig.from_json_file(json_file=cfg.CKPT.BERT.CFG)

    return BertModel(
        config=bert_cfg, add_pooling_layer=cfg.MODEL.BERT.ADD_POOLING
    )
