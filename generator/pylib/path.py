import datetime
import fnmatch
import os
import glob as _glob
import sys
import re 
import math
from pathlib import Path 

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def add_path(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def split(path):
    """Return dir, name, ext."""
    dir, name_ext = os.path.split(path)
    name, ext = os.path.splitext(name_ext)
    return dir, name, ext


def directory(path):
    return split(path)[0]


def name(path):
    return split(path)[1]


def ext(path):
    return split(path)[2]


def name_ext(path):
    return ''.join(split(path)[1:])


def change_ext(path, ext):
    if ext[0] == '.':
        ext = ext[1:]
    return os.path.splitext(path)[0] + '.' + ext


asbpath = os.path.abspath


join = os.path.join


def prefix(path, prefixes, sep='-'):
    prefixes = prefixes if isinstance(prefixes, (list, tuple)) else [prefixes]
    dir, name, ext = split(path)
    return join(dir, sep.join(prefixes) + sep + name + ext)


def suffix(path, suffixes, sep='-'):
    suffixes = suffixes if isinstance(suffixes, (list, tuple)) else [suffixes]
    dir, name, ext = split(path)
    return join(dir, name + sep + sep.join(suffixes) + ext)


def prefix_now(path, fmt="%Y-%m-%d-%H:%M:%S", sep='-'):
    return prefix(path, prefixes=datetime.datetime.now().strftime(fmt), sep=sep)


def suffix_now(path, fmt="%Y-%m-%d-%H:%M:%S", sep='-'):
    return suffix(path, suffixes=datetime.datetime.now().strftime(fmt), sep=sep)


def glob(dir, pats=None, recursive=False):  # faster than match, python3 only
    #   pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
#    for pat in pats:
        #   matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    matches = _glob.glob(dir)
    import pudb; pu.db
    sorted_files = sorted(matches, key=get_order)

    return sorted_files

def match(dir, pats, recursive=False):  # slow
    pats = pats if isinstance(pats, (list, tuple)) else [pats]

    iterator = list(os.walk(dir))
    if not recursive:
        iterator = iterator[0:1]

    matches = []
    for pat in pats:
        for root, _, file_names in iterator:
            for file_name in fnmatch.filter(file_names, pat):
                matches.append(os.path.join(root, file_name))

    return matches
