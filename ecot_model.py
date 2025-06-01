from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import torch

model_id = "Embodied-CoT/ecot-openvla-7b-bridge"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": 0},   # keep everything on cuda:0
    low_cpu_mem_usage=True     # optional; you can drop it
)

model.eval()
