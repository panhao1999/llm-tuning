# llm-tuning
该项目旨在学习模型微调原理，采用公开数据集 alpaca_gpt4_data_zh 对 Bloom-1.4B 模型进行微调并推理。微调技术包括：bitfit-tuning、prompt-tuning、p-tuning、prefix-tuning和lora-tuning。

数据集总数约5万条，下载地址：https://huggingface.co/datasets/shibing624/alpaca-zh

模型文件下载地址：https://huggingface.co/Langboat/bloom-1b4-zh

# Bloom模型介绍
Bloom 模型以 Transformer 架构为基础构建，Transformer 架构的自注意力机制能让模型有效处理序列数据中的长距离依赖关系，在自然语言处理任务中表现卓越。Bloom 模型采用了与 GPT 系列相似的解码器（Decoder）架构，使其在生成式任务上具有出色性能。

# 高效微调技术
在传统的预训练语言模型微调中，通常需要对模型的所有参数进行更新，对存储和计算上要求较高，特别是对于参数规模巨大的现代语言模型。而高效微调技术，旨在显著减少微调过程中的内存需求，同时保持模型性能。

1. BitFit-tuning
   在神经网络中，每一层通常由线性变换（权重矩阵与输入相乘）和偏置项（一个可学习的向量，与线性变换结果相加）组成。传统微调会更新权重矩阵和偏置项，但 BitFit-tuning 只调整偏置项。例如，在一个简单的全连接层中，线性变换为 wx+b，其中w是权重矩阵，x是输入，b是偏置向量。BitFit-tuning 仅对b进行更新，其余参数保持冻结。偏置参数在神经网络中虽然数量相对较少，但在模型的预测中起着关键作用，这种方法在保持模型性能的同时，极大地降低了微调成本。
   
2. prompt-tuning
   Prompt-tuning 旨在设计和优化输入到预训练语言模型的文本提示（prompts），这些提示能够引导模型执行特定任务，比如文本分类、问答等。通过精心构建和微调这些提示，模型可以在不改变自身参数的情况下，适应不同的下游任务。使用 prompt 微调模型，只需要在训练数据前加入一小段 prompt ，在模型创建时训练 prompt 的表示层（embedding）。prompt 分为：hard prompt（指定具体任务）、soft prompt（随机初始化，让模型自己学习是任务）。
   
3. p-tuning
   在 prompt-tuning 的基础上，对prompt部分进一步编码计算，加速收敛。其中，peft支持两种编码方式：LSTM、MLP，而 prompt 形式只有 soft prompt。
   
4. prefix-tuning
   prompt-tuning 和 p-tuning 是将 prompt 进行编码，然后和输入数据的编码进行拼接操作，再一块传入 transformer 结构。但是，prefix-tuning 不是在输入层进行拼接，而是以 past_key_values 的形式将可学习部分放入整个 transformer 中每一层，即：在 q、k、v 计算中的 key 和 value 前面加入前缀 past_key 和 past_value。
   
5. lora-tuning
   lora-tuning 通过矩阵分解的方式，将原本要跟新的大参数矩阵分解为两个小矩阵。即 W = W + W' = W + BA。实际操作中，需要在矩阵计算中增加一个旁系分支，旁系分支由两个低秩矩阵A和B组成。训练完成后，可以和原始模型的权重进行合并，没有额外推理开销。

# 操作步骤
1. 将数据集加载到根目录下的data文件中；
   
2. 执行每个微调任务的代码，其中包含了数据加载、数据预处理、模型加载、微调参数常见、模型创建和输出。

3. 在推理阶段，将原始模型文件与微调监测点进行合并，执行推理任务。


