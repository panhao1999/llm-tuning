'''
使用bitfit微调模型，wx+b，即只微调模型中所有的bias参数部分
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset, load_dataset

dataset = load_dataset("json", data_files="data/alpaca_gpt4_data_zh.json")
ds = dataset["train"]
print(ds)

tokenizer = AutoTokenizer.from_pretrained("../bloom-1b4-zh")
print(tokenizer)

"""
Given a list of examples, process them into a format that can be fed into the model.
"""
def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() +
                            "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]  # 只对response进行计算loss
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
# print(tokenized_ds[0])

print("input_ids: ", tokenizer.decode(tokenized_ds[1]["input_ids"]))
print("attention_mask: ", tokenizer.decode(tokenized_ds[1]["attention_mask"]))
print("labels: ", tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"]))))

# 加载模型
model = AutoModelForCausalLM.from_pretrained("../bloom-1b4-zh")

print("num_param_in_model: ", sum(p.numel() for p in model.parameters()))  # 打印参数量

# 只训练bias参数
num_param = 0
for name, param in model.named_parameters():
    if "bias" not in name:
        param.requires_grad = False
    else:
        num_param += param.numel()
print("num_param: ", num_param)

# p配置训练参数
args = TrainingArguments(
    output_dir="../bitfit_tuning/results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=50,
    num_train_epochs=1,
    save_steps=10000,
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

trainer.train()


# 推理
model = model.cuda()
ipt = tokenizer("\n".join(["Human: " + "考试有哪些技巧？", ""]).strip() + "\n\nAssistant: ",
                return_tensors="pt").to(model.device)
encode_res = model.generate(**ipt, max_new_tokens=128, do_sample=True)[0]
res = tokenizer.decode(encode_res, skip_special_tokens=True)
print(res)