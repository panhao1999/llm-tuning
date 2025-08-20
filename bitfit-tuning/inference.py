from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ckpt_dir = "../bitfit_tuning/results/checkpoint-20000"

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ckpt_dir,
    device_map="auto",           # 多卡/CPU 自动
    trust_remote_code=True
)
ipt = tokenizer("\n".join(["Human: " + "考试有哪些技巧？", ""]).strip() + "\n\nAssistant:",
                return_tensors="pt").to(model.device)
encode_res = model.generate(**ipt, max_length=512, do_sample=True, temperature=0.7)[0]
res = tokenizer.decode(encode_res, skip_special_tokens=True)
print(res)