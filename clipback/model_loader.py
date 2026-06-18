import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from peft import PeftModel
from clipback.config import MODEL_NAME, LORA_DIR, DEVICE


def load_models(mode: str):
    """
    mode: "baseline" | "lora" | "ensemble"
    반환: (model_or_tuple, processor)
    """
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    if mode == "ensemble":
        model_base = _load_base(processor)
        model_lora = _load_lora(processor)
        return (model_base, model_lora), processor

    if mode == "lora":
        return _load_lora(processor), processor

    return _load_base(processor), processor


def _load_base(processor):
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return model


def _load_lora(processor):
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    model.text_model = PeftModel.from_pretrained(model.text_model, LORA_DIR)
    model.text_model = model.text_model.merge_and_unload()
    proj_state = torch.load(
        f"{LORA_DIR}/text_projection.pt",
        map_location=DEVICE,
        weights_only=True
    )
    model.text_projection.load_state_dict(proj_state)
    return model


def get_text_embedding(model, processor, query: str) -> "np.ndarray":
    """텍스트 쿼리 → L2 정규화된 numpy 벡터 (앙상블 자동 처리)"""
    import numpy as np

    if isinstance(model, tuple):
        model_base, model_lora = model
        emb_base = _text_emb_single(model_base, processor, query)
        emb_lora = _text_emb_single(model_lora, processor, query)
        emb = (emb_base + emb_lora) / 2
        return (emb / np.linalg.norm(emb)).astype("float32")

    return _text_emb_single(model, processor, query)


def _text_emb_single(model, processor, query: str):
    import numpy as np
    inputs = processor(
        text=[query], return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(DEVICE)
    with torch.inference_mode():
        out = model.text_model(**inputs)
        emb = model.text_projection(out.pooler_output)
        emb = F.normalize(emb, dim=-1)
    return emb.cpu().numpy()[0]