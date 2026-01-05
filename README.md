# CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation

This repository contains the implementation of **CollabCoder**, a framework for improving large language model (LLM)-based code generation through **dynamic planning** and **self-improving debugging**.

## 📊 Benchmarks

We evaluate CollabCoder on four widely used code generation benchmarks:

* **HumanEval**
* **MBPP**
* **XCodeEval**
* **LiveCodeBench**

👉 For **LiveCodeBench**, please manually download the dataset here:
[Link](https://figshare.com/s/f450d9e553d0394f365b)

After downloading, place the dataset into:

```
data/LiveCodeBench
```

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-repo/CollabCoder.git
cd CollabCoder
pip install -r requirements.txt
```

## 🚀 Usage

To run CollabCoder on a specific dataset, use the following command:

```bash
python src/main.py --model GPT4oMini --dataset HumanEval --strategy CollabCoder
```

* Replace `--dataset HumanEval` with one of:
  `HumanEval`, `MBPP`, `XCodeEval`, `LiveCodeBench`
* Replace `--model GPT4oMini` with your preferred model.

### Example Runs

```bash
# Run on MBPP
python src/main.py --model GPT4oMini --dataset MBPP --strategy CollabCoder

# Run on LiveCodeBench
python src/main.py --model GPT4oMini --dataset LCB --strategy CollabCoder
```
