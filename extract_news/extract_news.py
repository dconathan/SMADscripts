#!/usr/bin/python
"""
doc string
"""
import re
import datetime as dt
import os
import sys
import json


def main(data_in, data_out):

    if os.path.isdir(data_in):
        data = extract_news_dir(data_in)
    elif os.path.isfile(data_in):
        data = extract_news_file(data_in)
    else:
        print("{} doesn't appear to be a valid file or directory".format(data_in))
        sys.exit(1)

    with open(data_out, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


DEBUG = False

SPLITTER_RE = re.compile(r'\d+ of \d+ DOCUMENTS')
WHITESPACE_RE = re.compile(r'\s+')
LINES_RE = re.compile(r'.*(\r?\n)+')
ONE_LINE_RE = re.compile(r'.*\r?\n')
HEADLINE_RE = re.compile(r'(.*\r?\n){2}')
EDITION_RE = re.compile(r'.*edition.*', re.IGNORECASE)
SINGLE_NEWLINE_RE = re.compile(r'\r?\n')
DOUBLE_NEWLINE_RE = re.compile(r'(\r?\n){2}')
TRIPLE_NEWLINE_RE = re.compile(r'(\r?\n){3}')
QUAD_NEWLINE_RE = re.compile(r'(\r?\n){4}')
DAY_RE = re.compile(r'(mon|tues|wednes|thurs|fri|satur|sun)day', re.IGNORECASE)
TIME_RE = re.compile(r'\d+:\d+( [AP]M \S*)?')
BYLINE_RE = re.compile(r'BYLINE:')
SECTION_RE = re.compile(r'SECTION:')
BLOG_RE = re.compile(r'\(.*\)')
LENGTH_RE = re.compile(r'LENGTH:')
DATELINE_RE = re.compile(r'DATELINE:')
LOADDATE_RE = re.compile(r'LOAD-DATE:')
LANGUAGE_RE = re.compile(r'LANGUAGE:')
DOCUMENTTYPE_RE = re.compile(r'DOCUMENT-TYPE:')
PUBLICATIONTYPE_RE = re.compile(r'PUBLICATION-TYPE:')
GRAPHIC_RE = re.compile(r'GRAPHIC:')
DISTRIBUTION_RE = re.compile(r'DISTRIBUTION:')
EMAIL_LINE_RE = re.compile(r'^\S+@\S+\s+$', re.MULTILINE)  # email address on line by itself
URL_RE = re.compile(r'URL:')
DATE_RE = re.compile(r'\S+ \d+, \d+ ' + DAY_RE.pattern + r'( \d+:\d+ [AP]M \S*)?', re.IGNORECASE)
HEADERFOOTER_RE = re.compile(r'( ){3}.*\r?\n')
META_RE = re.compile(r'[A-Z\-]+:.*\r?\n')
FIVE_OR_MORE_SPACES = re.compile(r'( ){5}[ ]*')


DATE_FORMAT = '%B %d, %Y'


def get_all_files(directory):
    assert os.path.isdir(directory), '{} is not a valid directory'.format(directory)
    return [os.path.abspath(os.path.join(directory, f)) for f in next(os.walk(directory))[2]]


def extract_news_dir(directory):
    filenames = get_all_files(directory)

    out = []

    for f in filenames:
        out += extract_news_file(f)

    print("Extracted {} articles".format(len(out)))

    return out


def extract_news_file(filename):

    with open(filename, 'rb') as f:
        raw_text = f.read().decode('utf-8-sig')

    return extract_news_string(raw_text, source=filename)


def extract_news_string(raw_text, source=''):
    raw_text = SPLITTER_RE.split(raw_text)

    raw_text = [s for s in raw_text if strip_regex(s, WHITESPACE_RE) != '']

    clean_data = []

    if DEBUG:
        raw_text = raw_text[:3]

    for i, rt in enumerate(raw_text):

        article_data = dict()

        article_data['source'] = os.path.split(source)[-1]
        article_data['source_index'] = i

        article = delete_all(rt, FIVE_OR_MORE_SPACES)

        article, meta = extract_and_delete_all(rt, META_RE)

        header_article_footer = [a.strip() for a in TRIPLE_NEWLINE_RE.split(article) if SINGLE_NEWLINE_RE.sub('', a)]

        header = header_article_footer[0]
        footer = header_article_footer[-1]
        article = '\n\n'.join(header_article_footer[1:-1])

        header = [h.strip() for h in SINGLE_NEWLINE_RE.split(header) if h]

        newspaper = header.pop(0)

        # This assumes the blog section is below newspaper, which it is in case of The New York Times Blogs!
        if 'Blogs' in newspaper:
            blog_section = header.pop(0)
        else:
            blog_section = ''

        date = header.pop(0)
        date = delete_regex(delete_regex(date, DAY_RE), TIME_RE).strip()
        date = dt.datetime.strptime(date, DATE_FORMAT).date()
        article_data['date'] = date.isoformat()

        header = '\n'.join(header)

        article_data['newspaper'] = newspaper.strip()

        header, edition = extract_and_delete_regex(header, EDITION_RE)
        article_data['edition'] = edition.strip()

        article_data['blog_section'] = blog_section.strip()

        meta, article_data['byline'] = extract_meta(meta, BYLINE_RE)
        meta, article_data['length'] = extract_meta(meta, LENGTH_RE)
        meta, article_data['section'] = extract_meta(meta, SECTION_RE)
        meta, article_data['dateline'] = extract_meta(meta, SECTION_RE)

        meta, article_data['url'] = extract_meta(meta, URL_RE)
        meta, article_data['load-date'] = extract_meta(meta, LOADDATE_RE)
        meta, article_data['language'] = extract_meta(meta, LANGUAGE_RE)
        meta, article_data['distribution'] = extract_meta(meta, DISTRIBUTION_RE)
        meta, article_data['publication-type'] = extract_meta(meta, PUBLICATIONTYPE_RE)
        meta, article_data['graphic'] = extract_meta(meta, GRAPHIC_RE)

        article = article.strip()

        headline = header.strip()

        article_data['headline'] = headline.strip()

        article = delete_all(article + '\n', EMAIL_LINE_RE)

        article_data['article'] = clean_article(article)
        article_data['paragraphs'] = article_data['article'].split('\n\n')

        clean_data.append(article_data)

    return clean_data


def strip_regex(string, regex):
    # strip beginning
    if string:
        while regex.match(string[0]):
            string = string[1:]
            if string == '':
                break
    # strip end
    if string:
        while regex.match(string[-1]):
            string = string[:-1]
            if string == '':
                break

    return string


def extract_regex(string, regex, head=False):
    """Returns first match of regex or empty string if not found"""
    search = regex.search(string)
    if search:
        if head:
            return string[:search.end()]
        else:
            return search.group(0)
    else:
        return ''


def delete_regex(string, regex, head=False):
    """Deletes first match of regex or unchanged original if not found"""
    search = regex.search(string)
    if search:
        if head:
            return string[search.end():]
        else:
            return string[:search.start()] + string[search.end():]
    else:
        return string


def extract_and_delete_regex(string, regex, head=False):
    extracted = extract_regex(string, regex, head)
    new_string = delete_regex(string, regex, head)

    return new_string.strip(' '), extracted


def extract_meta(string, regex):
    start = regex.search(string)
    if start:
        substring = string[start.start():]
        string = string[:start.start()]
        end = META_RE.search(substring)
        if end:
            string += substring[end.end():]
            substring = substring[:end.end()]
        substring = delete_regex(substring, regex)
        substring = ''.join(substring.splitlines()).strip()
    else:
        substring = ''
    return string, substring


def extract_paragraph(string):
    out = ''
    i = 0
    string, p = extract_and_delete_regex(string, ONE_LINE_RE)
    return p.strip(), string

    ''' 
        out += temp
        print(i, temp)
        i += 1
        if temp == '' or not string or string.replace('\r', '')[0] == '\n':
            break
    print()
    print(out)
    print()
    return '\n\n'.join(out.splitlines()).strip(), string
    '''


def clean_article(article):
    article = (article.strip() + '\n').replace("''", '"')
    paragraphs = []
    while True:
        p, article = extract_paragraph(article)
        if p != '':
            paragraphs.append(p)
        if article.strip() == '':
            break
    return '\n\n'.join(paragraphs)


def extract_all(string, regex):
    out = ''
    while True:
        string, temp = extract_and_delete_regex(string, regex)
        if temp == '':
            break
        out += temp
    return out


def delete_all(string, regex):
    while True:
        string, temp = extract_and_delete_regex(string, regex)
        if temp == '':
            break
    return string


def extract_and_delete_all(string, regex):
    return delete_all(string, regex), extract_all(string, regex)


def _test():
    assert strip_regex(' \t\t \n\r  hello world\r \n \t\t  ', WHITESPACE_RE) == 'hello world'
    assert strip_regex(' \t\t \n  \r\n \t\t  ', WHITESPACE_RE) == ''
    assert extract_regex('foobar \t\t \n\r  hello world\r \n \t\t  ', WHITESPACE_RE) == ' \t\t \n\r  '
    assert extract_regex('foobar \t\t \n\r  hello world\r \n \t\t  ', WHITESPACE_RE, True) == 'foobar \t\t \n\r  '
    assert extract_regex('helloworld', WHITESPACE_RE) == ''
    assert delete_regex('hello \n\nworld', WHITESPACE_RE) == 'helloworld'
    assert delete_regex('hello \n\nworld', WHITESPACE_RE, True) == 'world'
    assert delete_regex('helloworld', WHITESPACE_RE) == 'helloworld'


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("\nUsage: \n\tpython extract_news.py INFILE OUTFILE\n\n\t"
              "INFILE:\t\tfile or directory from which to extract\n\tOUTFILE:\tjson file to save to\n\nExample:"
              "\n\tpython extract_news.py examples/ out.json\n")
    else:
        main(sys.argv[1], sys.argv[2])
