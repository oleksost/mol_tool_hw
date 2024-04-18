## Installation

```bash
conda create --name my_env python=3.9
conda activate my_env
git clone https://github.com/oleksost/mol_tool_hw.git
cd mol_tool_hw
touch .env
echo OPENAI_API_KEY=\"YOUR OPEN AI KEY\" >> .env
pip install -e .
```

## Usage
See [demo.ipynb](https://github.com/oleksost/mol_tool_hw/blob/main/demo.ipynb) for a demo.

## Functionality:

### Part 1: Custom Retriever Tool Development

Retriever tool is implemented using LangChain. The retriever tool can access and parse the content of the specified paper. The tool can handle different types of queries related to the paper's content, such as summarization, specific data.

The retrieval tool is implemented in [src/mol_tools/tools.py](https://github.com/oleksost/mol_tool_hw/blob/dec1d1578ff7fbca3cd2190a79cd293c4e9e20c5/src/mol_tools/tools.py#L116).

A custom text filter for discovering SMILES strings is implemented in [src/mol_tools/tools.py:195](https://github.com/oleksost/mol_tool_hw/blob/dec1d1578ff7fbca3cd2190a79cd293c4e9e20c5/src/mol_tools/tools.py#L195).



### Part 2: Predictive Machine Learning Model Tooling Implementation

This projet supports training a regression model to predict hydration free energy for SMILE molecules using FreeSolvdataset. It deploys the trained model into a custom LangChain tool. The custom LangaChain tool is implemented under [src/mol_tools/tools.py](https://github.com/oleksost/mol_tool_hw/blob/dec1d1578ff7fbca3cd2190a79cd293c4e9e20c5/src/mol_tools/tools.py#L25).

Currently the code supports 4 model types, all implemented using sklear library:
- linear regression
- tree based regressio model
- mlp regressor
- ensamble regressor (AutoSklearnRegressor)

All models are implemented in [src/mol_tools/models.py](https://github.com/oleksost/mol_tool_hw/blob/main/src/mol_tools/models.py).

**Adding model**: new models can added to [src/mol_tools/models.py](https://github.com/oleksost/mol_tool_hw/blob/main/src/mol_tools/models.py) file by subclassing the Model class. Use `register_model` decorator for each new model class.