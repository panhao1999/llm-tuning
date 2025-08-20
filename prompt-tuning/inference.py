# 加载训练好的peft模型
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "../bloom-1b4-zh"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("../bloom-1b4-zh")
# 要确保model和监测点在同一个设备上（CPU/GPU）
peft_model = PeftModel.from_pretrained(model, "../prompt_tuning/results/checkpoint-1500")
peft_model = peft_model.cuda()

ipt = tokenizer("\n".join(["Human: " + "考试有哪些技巧？", ""]).strip() + "\n\nAssistant:",
                return_tensors="pt").to(peft_model.device)
encode_res = peft_model.generate(**ipt, max_length=512, do_sample=True, temperature=0.7)[0]
res = tokenizer.decode(encode_res, skip_special_tokens=True)
print(res)