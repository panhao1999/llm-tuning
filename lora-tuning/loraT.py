'''
lora: 通过矩阵分解的方式，将原本要跟新的大矩阵分解为两个小矩阵。
W = W + W' = W + BA
即：在矩阵计算中增加一个旁系分支，旁系分支由两个低秩矩阵A和B组成。
训练完成后，可以和原始模型的权重进行合并，没有额外推理开销。
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
# 打印所有的可训练参数部分
for name, param in model.named_parameters():
    print(name)

# lora-tuning
# 配置文件
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense_4h_to_h"],  # 指定需要调参的部分
    modules_to_save=["word_embeddings"],  # 除了lora以外，指定其他训练参数
)
print("config: ", config)

# 创建模型
model = get_peft_model(model, config)
print("model: ", model)
print("num_param_in_model: ", model.print_trainable_parameters())  # 打印可训练参数量、模型总参数量、占比

# 配置训练参数
args = TrainingArguments(
    output_dir="../lora_tuning/results",
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