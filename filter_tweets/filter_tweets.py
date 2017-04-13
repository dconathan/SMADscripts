#!/usr/bin/python
"""
docstring for this script
"""
import os
import shutil
from itertools import takewhile
from itertools import repeat
import re
import time
import sys


def main(source, dest):

    filter_process(source, dest, compose(filter_out_exclusion_list,
                                         filter_in_english,
                                         filter_out_handles,
                                         filter_in_inclusion_list)
                   )


class LineReader:
    """
    This class iterates through a file line by line in a streaming fashion.
    It's useful if the file you are processing is very large and you don't want to load the whole thing into memory.
    
    Uses:
        line_reader = LineReader('verybigfile.txt')
        for line in line_reader:
            # do some processing on each line, like counting words or filtering, etc.
            
        # Or if you file isn't very big, you can just load the whole thing into memory by turning it into a list:
        x = list(line_reader)
        
        # len() is defined, so:
        len(line_reader)
        # will tell you how many lines are in the file without having to iterate over the whole thing
        # or load it into memory
    """
    def __init__(self, source):
        self.source = source
        self.length = None

    def __iter__(self):
        with open(self.source) as f:
            for line in f:
                yield line.replace('\n', '')

    def __len__(self):
        """ Counts the lines in self.source """
        if self.length is None:
            with open(self.source, 'rb') as f:
                bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
                self.length = sum(buf.count(b'\n') for buf in bufgen)
        return self.length


class LineWriter:
    """
    Designed as a counterpart to LineReader, LineWriter writes lines to a file one line at a time.
    
    Uses:
        line_writer = LineWriter('verybigfile_processed.txt')
        for line in LineReader('verybigfile.txt'):
            # do some processing on line
            line_writer.writeline(line)
            
    If the output file (e.g. 'verybigfile_processed.txt') already exists, a backup will be made
    and it will print a warning.
    However, if you run it again without renaming the backup, your backup will get overwritten so pay attention!
    Do not read and write to the same file - your data will disappear.
    """
    def __init__(self, dest):
        self.dest = dest
        if os.path.exists(self.dest):
            print("Warning!  The file {} already exists, making a backup called {}.backup "
                  "and overwriting original".format(dest, dest))
            shutil.copy(dest, dest + '.backup')
        with open(self.dest, 'w') as f:
            f.write('')

    def writeline(self, line):
        if line is not None:
            with open(self.dest, 'a') as f:
                f.write(line + '\n')


"""
The basic filter paradigm we are using is to define each filter as a function that looks at a string and returns either:
The unchanged string (indicating that it should not be filtered out), or
None (indicating that it should be filtered out)

If all your filter functions have the @composable decorator they will simply pass None through the filter chain
(and then they won't get written by LineWriter, which ignores None)

So then you build one big filter by composing a bunch of small filters using compose()

Simple example:

@composable
def no_meow(s):
    if 'meow' in s:
        return None
    else:
        return s


@composable
def no_cat(s):
    if 'cat' in s:
        return None
    else:
        return s
        
no_meow_or_cat = compose(no_meow, no_cat)
# no_meow_or_cat is now a function that returns None if the input contains meow or cat...
no_meow_or_cat('hello world')  # == 'hello world'
no_meow_or_cat('meow world')   # == None
"""


def compose(*args):
    def big_func(arg):
        for f in args:
            if arg is None:
                return None
            arg = f(arg)
        return arg
    return big_func


def filter_process(source, dest, filt):

    source_reader = LineReader(source)
    dest_writer = LineWriter(dest)

    total = len(source_reader)

    t0 = time.time()
    t1 = t0

    for i, line in enumerate(source_reader):
        dest_writer.writeline(filt(line))
        if time.time() > t1 + 5:
            t1 = time.time()
            print("{0:d}/{1:d} processed. "
                  "Estimated time remaining: {2:.1f}s".format(i + 1, len(source_reader),
                                                              (t1 - t0) / (i + 1) * (len(source_reader) - i)))


EXCLUSION_LIST = list(LineReader('exclusion_list.txt'))


def filter_out_exclusion_list(s):
    for word in EXCLUSION_LIST:
        if word.lower() in s.lower():
            return None
    return s

ENGLISH_WORDS = set(LineReader('english_words.txt'))
ENGLISH_THRESHOLD = 5


def filter_in_english(s):
    n_words = 0
    # '\x01' is the hive default separator, which .split doesn't split on
    for word in s.lower().replace('\x01', ' ').replace('"', ' ').split():
        if word[-1] == 's':
            word = word[:-1]
        if word in ENGLISH_WORDS:
            n_words += 1
            if n_words == ENGLISH_THRESHOLD:
                return s
    return None


HANDLES_RE = re.compile(r'@\w*(gun|shoot|firearm)')

def filter_out_handles(s):
    if bool(HANDLES_RE.search(s.lower())):
        return None
    else:
        return s


INCLUSION_LIST = list(LineReader('inclusion_list.txt'))
INCLUSION_LIST = [word.lower().strip() + 's?' for word in INCLUSION_LIST]
INCLUSION_RE = re.compile(r'\b(' + '|'.join(INCLUSION_LIST) + r')\b')

def filter_in_inclusion_list(s):
    if bool(INCLUSION_RE.search(s.lower())):
        return s
    else:
        return None


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("\nUsage: \n\tpython filter_tweets.py INFILE OUTFILE\n\n\t"
              "INFILE:\t\tsource file from which to read and filter\n\tOUTFILE:\tfile to write to\n\nExample:"
              "\n\tpython filter_tweets.py test_in.csv test_out.csv\n")
    else:
        main(sys.argv[1], sys.argv[2])

