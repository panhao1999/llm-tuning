from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained("../bloom-1b4-zh")
tokenizer = AutoTokenizer.from_pretrained("../bloom-1b4-zh")

# 加载lora模型
p_model = PeftModel.from_pretrained(model, "../lora_tuning/results/checkpoint-1000")
print("p_model: ", p_model)
print("num_param_in_model: ", sum(p.numel() for p in p_model.parameters()))

p_model = p_model.cuda()
ipt = tokenizer("\n".join(["Human: " + "考试有哪些技巧？", ""]).strip() + "\n\nAssistant:",
                return_tensors="pt").to(p_model.device)
encode_res = p_model.generate(**ipt, max_length=512, do_sample=True, temperature=0.7)[0]
res = tokenizer.decode(encode_res, skip_special_tokens=True)
print(res)

# # 权重合并
# merge_model = p_model.merge_and_unload()
#
# merge_model = merge_model.cuda()
# ipt = tokenizer("\n".join(["Human: " + "考试有哪些技巧？", ""]).strip() + "\n\nAssistant:",
#                 return_tensors="pt").to(merge_model.device)
# encode_res = merge_model.generate(**ipt, max_length=256, do_sample=True, temperature=0.7)[0]
# res = tokenizer.decode(encode_res, skip_special_tokens=True)
# print(res)
#
# 保存模型
# merge_model.save_pretrained("../prompt_tuning/results/merge_model")
