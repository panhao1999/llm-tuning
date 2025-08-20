'''
使用prompt微调模型，冻结主模型全部参数，在训练数据前加入一小段prompt，只训练prompt的表示层（embedding）。
prompt分为：hard prompt（指定具体任务）、soft prompt（随机初始化，让模型自己学习是任务）
注意：数据该怎样处理就怎样处理，prompt在模型创建的时候进行操作
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
'''
PromptTuningConfig, get_peft_model 配合得到 peft model
TaskType  指定任务类型
PromptTuningInit  指定 hard prompt 还是 soft prompt
'''
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit


dataset = load_dataset("json", data_files="../data/alpaca_gpt4_data_zh.json")
ds = dataset["train"]
print("ds: ", ds)

tokenizer = AutoTokenizer.from_pretrained("../bloom-1b4-zh")
print("tokenizer: ", tokenizer)

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

# (1) soft prompt 配置文件
# config = PromptTuningConfig(
#     task_type=TaskType.CAUSAL_LM,
#     num_virtual_tokens=10,  # prompt的token长度
# )

# (2) hard prompt 配置文件
config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                            prompt_tuning_init=PromptTuningInit.TEXT,
                            prompt_tuning_init_text="下面是人和智能体之间的对话。",
                            num_virtual_tokens=len(tokenizer("下面是人和智能体之间的对话。")["input_ids"]),
                            tokenizer_name_or_path="../bloom-1b4-zh",)

print("config: ", config)

# 创建模型
model = get_peft_model(model, config)
print("model: ", model)
print("num_param_in_model: ", model.print_trainable_parameters())  # 打印可训练参数量、模型总参数量、占比

# 配置训练参数
args = TrainingArguments(
    output_dir="../prompt_tuning/results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=50,
    num_train_epochs=1,
    save_steps=500,
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