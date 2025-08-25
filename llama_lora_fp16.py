from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
import torch

# 加载数据集
dataset = load_dataset("json", data_files="../data/alpaca_gpt4_data_zh.json")
ds = dataset["train"]
print("ds: ", ds)

#数据集预处理
tokenizer = AutoTokenizer.from_pretrained("C:/Users/16636/Desktop/modelFile/Llama-2-7b-ms")
tokenizer.padding_side = "right"  # 当batch_size>1时，需要设置padding_side="right"，否则可能会不收敛

tokenizer.pad_token_id = 2
print("tokenizer: ", tokenizer)


def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() +
                            "\n\nAssistant: ", add_special_tokens=False)
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]  # 只对response进行计算loss
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

print(tokenized_ds[0]["input_ids"])
print(tokenizer.eos_token_id, tokenizer.eos_token)
print("input_ids: ", tokenizer.decode(tokenized_ds[0]["input_ids"]))
print("attention_mask: ", tokenizer.decode(tokenized_ds[0]["attention_mask"]))
print("labels: ", tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[0]["labels"]))))

# print(tokenizer("呀", add_special_tokens=False))

# 创建模型
model = AutoModelForCausalLM.from_pretrained("C:/Users/16636/Desktop/modelFile/Llama-2-7b-ms",
                                             torch_dtype=torch.half)
model = model.to("cuda")

print("num_param_in_model: ", sum(p.numel() for p in model.parameters()))  # 打印参数量

# # 打印所有的可训练参数部分
# for name, param in model.named_parameters():
#     print(name, param.dtype)

# lora-tuning
# 配置文件
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(task_type=TaskType.CAUSAL_LM)
print("config: ", config)

# 创建模型
model = get_peft_model(model, config)
model.enable_input_require_grads()
model.half()

print("model: ", model)
print("num_param_in_model: ", model.print_trainable_parameters())  # 打印可训练参数量、模型总参数量、占比

# 配置训练参数
args = TrainingArguments(
    output_dir="../fp16_lora_tuning/llama_results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
    save_steps=500,
    gradient_checkpointing=True,  # 启用gradient checkpointing，减少内存占用
    adam_epsilon=1e-4,
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds.select(range(1000)),
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

trainer.train()