# llm-tuning

本项目聚焦于深入探究模型微调原理，选用公开数据集 alpaca_gpt4_data_zh，对 Bloom-1.4B 、Llama-2-7B-ms 模型开展微调训练与推理操作。项目运用的微调技术涵盖 BitFit-tuning、Prompt-tuning、P-tuning、Prefix-tuning 以及 Lora-tuning。

数据集规模约 5 万条，下载链接：https://huggingface.co/datasets/shibing624/alpaca-zh

Bloom 模型文件获取地址：https://huggingface.co/Langboat/bloom-1b4-zh

Llama2 模型文件获取地址：https://modelscope.cn/models/modelscope/Llama-2-7b-ms/files

# 高效微调

在传统的预训练语言模型微调中，通常需要对模型的所有参数进行更新，对存储和计算上要求较高，特别是对于参数规模巨大的现代语言模型。而高效微调技术，旨在显著减少微调过程中的内存需求，同时保持模型性能。

1. BitFit-tuning

   在神经网络中，每一层通常由线性变换（权重矩阵与输入相乘）和偏置项（一个可学习的向量，与线性变换结果相加）组成。传统微调会更新权重矩阵和偏置项，但 BitFit-tuning 只调整偏置项。例如，在一个简单的全连接层中，线性变换为 wx+b，其中w是权重矩阵，x是输入，b是偏置向量。BitFit-tuning 仅对b进行更新，其余参数保持冻结。偏置参数在神经网络中虽然数量相对较少，但在模型的预测中起着关键作用，这种方法在保持模型性能的同时，极大地降低了微调成本。
   
2. Prompt-tuning

   Prompt-tuning 旨在设计和优化输入到预训练语言模型的文本提示（prompts），这些提示能够引导模型执行特定任务，比如文本分类、问答等。通过精心构建和微调这些提示，模型可以在不改变自身参数的情况下，适应不同的下游任务。运用 prompt 微调模型时，仅需在训练数据前加入一小段 prompt ，并在模型创建时训练 prompt 的表示层（embedding）。prompt 分为：hard prompt（明确指定具体任务）、soft prompt（随机初始化，让模型自己学习任务）两类。
   
3. P-tuning

   P-tuning 是在 prompt-tuning 的基础上，对 prompt 部分进一步编码计算，以此加速收敛。其中，PEFT库支持两种编码方式：LSTM、MLP，且 prompt 形式仅为 soft prompt。
   
4. Prefix-tuning

   Prompt-tuning 和 P-tuning 是将 prompt 进行编码后，与输入数据的编码进行拼接操作，再一同传入 transformer 结构。与之不同的是，Prefix-tuning 并非在输入层进行拼接，而是以 past_key_values 的形式将可学习部分融入整个 transformer 的每一层，即在 q、k、v 计算中的 key 和 value 前面添加前缀 past_key 和 past_value。
   
5. Lora-tuning

   Lora-tuning 通过矩阵分解的方式，将原本需要跟新的大参数矩阵分解为两个小矩阵。即 W = W + W' = W + BA。实际操作时，需要在矩阵计算中增设一个旁系分支，该旁系分支由两个低秩矩阵A和B组成。训练完成后，可将其与原始模型的权重进行合并，且不会产生额外的推理开销。

# 半精度训练
采用 Lora-tuning 对Llama 模型进行微调训练时，为了进一步降低资源占用率，在训练阶段做了一些改变。

1. 在加载模型时，指定参数 torch_dtype 为半精度加载。
   
2. 在 lora 配置文件中设置参数 gradient_checkpointing，此时，只有模型中少数关键层的输出会被保存，其他层的输出在需要计算梯度时会根据保存的输入和参数重新计算。这样，在训练过程中，内存中不需要一直保留所有中间层的激活值，大大降低了内存需求，但相应的也会增加训练时间。
 
3. 为了对模型中间层的输出计算梯度。训练阶段采用 enable_input_require_grads 来设置相关张量的梯度计算。

注：
   
   · 当采用多批量进行训练时，会增加padding补齐操作，为了避免发生损失不收敛的情况，此时，可设置另一个参数 tokenizer.padding_side = "right"。

   · 使用半精度训练时，adam 优化器 可能会造成 数值下溢出，此时，可在配置文件中调整 adam_epsilon 的数值即可。

   · Llama2 模型分词器会将非单独存在的 eos_token 切分，因此，对于 eos_token 要单独处理，否则训练后的模型在预测时不知道何时停止。

   · 半精度训练时，正确加入 eos_token 后，要将 pad_token_id 也设置为 eos_token_id，否则模型无法收敛。 

# 8bit模型训练
量化是一种模型压缩的方法，使用低精度数据类型对模型权重或者激活值进行表示。该方式可有效降低模型加载的内存空间，减少计算成本。

若要保持计算精度，需要进行反量化操作，此时，反量化后的结果与原始数据会有一定差异，即量化误差。

为了降低量化误差，提高量化精度：使用更多的量化参数（scale_factor），为矩阵不同行或列设置单独的 scale_facotor，这种方式被称为vector-wise量化。

存在问题：

在量化阶段，数据中可能会出现一些离群值（即超出某个分布范围的值），这可能造成量化误差较大，影响模型性能。此时可采用混合精度量化进行操作，从输入的隐含状态中，拉取离群值，对离群值进行FP16（半精度）量化，非离群值进行INT8量化。

# 操作步骤指南
1. 核心环境配置

   · Python版本：3.11.13

   · Pytorch版本：2.2.1

   · Transformers版本：4.33.1

   · Peft版本：0.5.0

   · Accelerate版本：0.22.0

   · Bitsandbytes版本：0.41.1  （ https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels ）

3. 数据及模型准备

   将数据集下载至根目录下的data文件中，并下载 Bloom 模型文件。
   
4. 模型微调执行

   执行每个微调任务的代码，这些代码包含了数据加载、数据预处理、模型加载、微调参数配置、模型微调和结果输出等环节。

5. 推理阶段操作

   在推理阶段，将原始模型文件与微调监测点进行合并，执行推理任务。

