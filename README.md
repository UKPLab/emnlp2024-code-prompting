# Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs



## Project structure
### Scripts
* `boardgameqa_code_prompt.ipynb` -- This notebook runs `code prompts` on `BoardgameQA`
* `boardgameqa_text_prompt.ipynb` -- This notebook runs `text prompts` on `BoardgameQA`
* `conditionalqa_code_prompt.ipynb` -- This notebook runs `code prompts` on `ConditionalQA`
* `conditionalqa_text_prompt.ipynb` -- This notebook runs `text prompts` on `ConditionalQA`
* `sharc_code_prompt.ipynb` -- This notebook runs `code prompts` on `ShARC`
* `sharc_code_text_prompt.ipynb` -- This notebook runs `text prompts` on `ShARC`

### Backend
* `src` -- This folder contain the classes that define `text prompts` and `code prompts` for `ConditionalQA` and `BoardgameQA`, and `ShARC`
* `data` -- This folder contains the the training, dev, and ICL demonstrations used in the experiments (including ablations).
* `outputs` -- This folder contains all the prompts (inputs and outputs). It also includes the evaluation results of each prompt. 

## Requirements
* openai
* langchain
* scikit-learn

You also need an Azure OpenAI or OpenAI API account and put your key in the notebook to run them.

## Installation
```
conda create --name code_prompting python=3.9
conda activate code_prompting
pip install -r requirements.txt
```

## Running the experiments 🏃
To reproduce our main experiments, you just need to run these notebooks:
* `boardgameqa_code_prompt.ipynb`
* `boardgameqa_text_prompt.ipynb`
* `conditionalqa_code_prompt.ipynb`
* `conditionalqa_text_prompt.ipynb`
* `sharc_code_prompt_prompt.ipynb`
* `sharc_code_txt_prompt.ipynb` 

❗️ Dont' forget to add your OpenAI API keys!
