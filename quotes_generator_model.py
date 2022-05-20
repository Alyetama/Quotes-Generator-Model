#!/usr/bin/env python
# coding: utf-8

import json
from pathlib import Path

import gpt_2_simple as gpt2
import language_tool_python
import numpy as np
from cleantext import clean
from greads import greads


def get_author_quotes(author_name,
                      output_file='quotes.json',
                      enable_multiprocessing=False,
                      language='en'):
    quotes = greads.scrape(author=author_name,
                           output_file=output_file,
                           enable_multiprocessing=enable_multiprocessing,
                           language=language)
    return quotes


def download_model(model_name='124M'):
    gpt2.download_gpt2(model_name=model_name)
    return


def preprocess_data(quotes,
                    author,
                    topic='general',
                    output_file='processed_quotes.txt'):
    bs = 64
    np.random.seed(1)

    if isinstance(quotes, str):
        if Path(quotes).exists():
            with open(quotes) as j:
                quotes = json.load(j)
    else:
        assert isinstance(quotes, list)

    with open(output_file, 'w') as f:
        for q in quotes:
            f.write(f"_TOPIC_ {topic} _QUOTE_ {q} _AUTHOR_ {author} _END_\n")

    encoded = gpt2.encode_dataset(output_file)
    return


def train(model_name='124M',
          steps=3000,
          restore_from='fresh',
          run_name='run0',
          print_every=1,
          sample_every=200,
          save_every=500):
    sess = gpt2.start_tf_sess()

    gpt2.finetune(sess,
                  dataset='text_encoded.npz',
                  model_name=model_name,
                  steps=steps,
                  restore_from=restore_from,
                  run_name=run_name,
                  print_every=print_every,
                  sample_every=sample_every,
                  save_every=save_every)


def train_pipeline(author_name):
    quotes_list = get_author_quotes(author_name)
    download_model()
    preprocess_data(quotes_list, author_name)
    train()


def generate(checkpoint_folder_path,
             topic='general',
             length=100,
             temperature=0.7,
             nsamples=10,
             batch_size=10,
             prefix=None):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=checkpoint_folder_path)
    results = gpt2.generate(sess,
                            run_name=checkpoint_folder_path,
                            prefix=f'_TOPIC_ {topic} _QUOTE_',
                            truncate='_END_',
                            length=length,
                            temperature=temperature,
                            nsamples=nsamples,
                            batch_size=batch_size,
                            return_as_list=True)
    return results


def clean_generated(results):
    tool = language_tool_python.LanguageTool('en-US')
    corrected_results = []
    for q in results:
        q = q.split('_TOPIC_ general _QUOTE_ ')[1].strip()
        if not q:
            continue
        if not q.endswith('.'):
            q = '.'.join(q.split('.')[:-1]) + "."

        matches = tool.check(q)
        corrected = language_tool_python.utils.correct(q, matches)
        corrected_clean = clean(q,
                                fix_unicode=True,
                                to_ascii=True,
                                lower=False,
                                no_line_breaks=False,
                                no_urls=True,
                                no_emails=True,
                                no_phone_numbers=True,
                                no_numbers=False,
                                no_digits=True,
                                no_currency_symbols=True,
                                no_punct=False,
                                lang='en')
        corrected_results.append(corrected_clean)

    tool.close()
    return corrected_results
