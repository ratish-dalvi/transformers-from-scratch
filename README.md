# Transformer Implementation From Scratch

A simple, beginner-friendly implementation of Transformers built from the ground up. It focuses on the core concepts of Transformers, simplifying and abstracting away extra code related to data preparation and training. Surprisingly, this code is quite effective- I could create a model on par with GPT2 for a training cost of less than $200 on Lambda Labs.


_There are so many implementations of Transformers? What makes this one special?_

Frankly, nothing. It's just a personal project to try and build a GPT model by myself. I've noticed that many other projects say their main code is really short, like less than 200 lines, but it's usually mixed with a lot of complicated code for handling data, training loops, and optimizers. I just want to concentrate on the main structure of the model, so I'm using tools from Hugging Face to handle everything else. As a result, the complete code needed to train a model comparable to OpenAI's GPT2 (with a loss of around 2.9 on OpenWebText) is less than 300 lines, comments included.

It consists of two main files:

- `model.py`: Implementation of a decoder-only transformer architecture in pytorch
- `gpt_trainer.py`: Data loading, processing, and model training using Hugging Face libraries

  
### Installation

I recommend setting up a new, dedicated environment for this project. 

```
git clone https://github.com/ratish-dalvi/transformers-from-scratch.git
pip install -r requirements
```

### Usage

To kick off a small model training job on a small fraction of the OpenWebText dataset (1%), run:
```
python3.9 gpt_trainer.py \
    --dataset_percent=1 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --eval_steps=500 \
    --logging_steps=50
```

To train a GPT-like model on the full OpenWebText dataset (read the next section before running) with default settings, run:
```
python gpt_trainer.py
```

To see the full list of defaults used in training, look at the top of gpt_trainer.py or run

```
python gpt_trainer.py --help
```

### Training Process

- **Data Download**:  
OpenAI trained GPT2 on the OpenWebText dataset, which was not publicly released. This repo uses an open-source version of the dataset available on Hugging Face, although it is important to note that this replica is not identical to the original dataset. This dataset is quite large, approximately 55 GB, and includes about 8 million texts. Make sure you have sufficient disk space before training. 
Initially, downloading the dataset (55 GB) may take some time, depending on your internet speed; after which it is cached locally and subsequent runs take seconds.

- **Tokenization**:  
Once the data is downloaded, the next step is tokenization. Its speed depends on your CPU cores and volume of data. For the full dataset, expect this to take between 30 minutes to 2 hours the first time (cached for later). Start with a smaller dataset portion (for example, dataset_percent=1) and scale up to the full dataset when you have GPUs. The trainer also divides the data into training and test sets, a process that takes about 5-10 minutes on the first run but is cached thereafter.

- **Training**:  
I used my personal 3090 GPU (24 GB VRAM) for training at first. With a setup similar to gpt2 (around 140 million parameters), a batch size of 16, and gradient accumulation, it reached a loss of 3.2 in two days. My estimate was that it would reach GPT2-like performance in about a week, which was too long. To speed up the process, I switched to an 8 x 16 GB V100 machine, which costs $4.4 per hour on Lambda Labs. This got me an evaluation loss of 2.9 in two days. For comparison, Karpathy's nanoGPT gets to that eval loss over 4 days using 8 A100 GPUs. Interestingly, my code does it for a quarter of the cost, which might be due to a lower evaluation frequency or perhaps some magic in the Hugging Face trainer. Because there is no way my code is better than the collective efforts of Karpathy and his open-source friends.

### Track Model Progress

Run
```
tensorboard --logdir=logs
```
and open `http://localhost:6006/` in your browser


### References

Code:

https://github.com/huggingface/transformers  
https://github.com/karpathy/minGPT  
https://github.com/karpathy/nanoGPT  
https://github.com/pbloem/former  
https://github.com/affjljoo3581/GPT2  

