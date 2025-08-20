'''
prompt-tuning和p-tuning是将prompt进行编码，然后和输入数据的编码进行拼接操作，然后一块传入transformer结构
而prefix-tuning: 不是在输入层进行拼接，而是以past_key_values的形式将可学习部分放入整个transformer中每一层。
即qkv计算中的key和value前面加入前缀past_key和past_value。
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset, load_dataset

# 加载数据集
dataset = load_dataset("json", data_files="../data/alpaca_gpt4_data_zh.json")
ds = dataset["train"]
print("ds: ", ds)

#数据集预处理
tokenizer = AutoTokenizer.from_pretrained("../bloom-1b4-zh")
print("tokenizer: ", tokenizer)

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

# 创建模型
model = AutoModelForCausalLM.from_pretrained("../bloom-1b4-zh")
print("num_param_in_model: ", sum(p.numel() for p in model.parameters()))  # 打印参数量

# prefix-tuning
# 配置文件
from peft import PrefixTuningConfig, get_peft_model, TaskType
config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,  # prompt的token长度
    prefix_projection=True,  # 是否对prefix进行线性映射
)
print("config: ", config)

# 创建模型
model = get_peft_model(model, config)
print("model: ", model)
print("num_param_in_model: ", model.print_trainable_parameters())  # 打印可训练参数量、模型总参数量、占比

# 配置训练参数
args = TrainingArguments(
    output_dir="../prefix_tuning/results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=10,
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