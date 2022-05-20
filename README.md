# Quotes Generator Model

Create a model that generate quotes of any popular author with two lines of code


## TL;DR

Use the Colab notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/Alyetama/51e07efbe3fc3cfdbf65523734ea9b2d/quotes-generator-model.ipynb)


---

## Getting Started

```sh
git clone https://github.com/Alyetama/Quotes-Generator-Model.git
cd Quotes-Generator-Model
pip install -r requirements.txt
```

## Usage


### Training a model:

```python

import quotes_generator_model as qgm

qgm.download_model(model_name='124M')

author_name = 'Oscar Wilde'

quotes = qgm.get_author_quotes(author_name)
qgm.preprocess_data(quotes, author_name)

qgm.train(model_name='124M',
          steps=3000,
          restore_from='fresh',
          run_name='run0',
          print_every=10,
          sample_every=100,
          save_every=300)
```

- Or you can simply run:

```python
import quotes_generator_model as qgm
qgm.train_pipeline('Oscar Wilde')
```

### Using a checkpoint to generate quotes:

```python
import quotes_generator_model as qgm

results = qgm.generate(checkpoint_folder_path='checkpoint/run0', nsamples=10)


# Optional: to clean the generated results (remove prefix, grammar checks, etc.):

clean_results = qgm.clean_generated(results)
```
