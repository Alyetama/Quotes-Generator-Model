# Quotes Generator Model

Create a model that generate quotes of any popular author with two lines of code


## TL;DR

```sh
git clone https://github.com/Alyetama/Quotes-Generator-Model.git && \
    cd Quotes-Generator-Model && \
    pip install -r requirements.txt
```

```python
import quotes_generator_model as qgm

qgm.train_pipeline('Oscar Wilde')
```

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

download_model(model_name='124M')

author_name = 'Oscar Wilde'

quotes = qdm.get_author_quotes(author_name)
preprocess_data(quotes, author_name)

train(model_name='124M',
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

results = generate(checkpoint_folder_path='checkpoint/run0', nsamples=10)


# Optional: to clean the generated results (remove prefix, grammar checks, etc.):

clean_results = clean_generated(results)
```
