# smad

This is a bunch of (mostly NLP) tools that I use regularly enough to make into my own module.  It is mostly for personal use so no guarantees.

It's mostly based on NLTK and regular expressions with some scipy/sklearn/etc. thrown in.

This module parallelizes a bunch of commonplace functions, like tokenizing.

The most useful thing is the `smad.process` function.  It allows you to chain together a bunch of functions and write the output to a file.

For example:

```python
smad.process('test.txt', [smad.split_by_sentence, smad.filter_out_short, smad.tokenize, smad.join_lines])
```

This should take some raw text in `test.txt`, and produce a file `test.txt.processed` which has each sentence from the
original file (longer than 5 words) tokenized on its own line.  If the output of the smad.process chain is not a string (e.g. a list of tokens),
it will json-serialize it.