#!/usr/bin/python

# test/demo source code for DMV/CCM  (Klein&Manning 2004)
# and its implementation by Franco M. Luque
# see: http://www.cs.famaf.unc.edu.ar/~francolq/en/proyectos/dmvccm

# Adjust corpus directory to point to "wsj" (or another corpus).
# must contain trees in MRG form.
# Exact directory is important.
# delete all other files (e.g. READMEs) from this directory.
# David Reitter, Aug 2013, reitter@psu.edu

CORPUS_DIR = "/Users/dr/Downloads/DMV/wsj_comb/parsed/mrg/wsj"
CORPUS_DIR = "/home/xtof/corpus/WSJ/wsj_comb"

import nltk
import nltk.corpus

from nltk.corpus import ptb


from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *


dependency_treebank = LazyCorpusLoader(
     'dependency_treebank', DependencyCorpusReader, '.*\.dp',
     encoding='ascii')

# from dmvccm import dmv
# m = dmv.DMV()
# m.train(2)
# s = 'DT NNP NN VBD DT VBZ DT JJ NN'.split()
# (t, p) = m.dep_parse(s)
# t.draw()
  

from dmvccm import ccm
from dmvccm.dmvccm import *

import wsj10
# tb = wsj10.WSJ10(basedir='wsj_comb/parsed/mrg')
# m = ccm.CCM(tb)
# print m

# m.train(2)

# m.train(10)
# m.test()


# can't do - presumably not tagged
#m = ccm.CCM(dependency_treebank)

#from nltk.corpus import treebank
#t = treebank.parsed_sents('wsj_0001.mrg')[0]


tb = wsj10.WSJ10(basedir=CORPUS_DIR)
#print tb.get_trees()
#print tb.raw()
tb.print_stats()

m = ccm.CCM(tb)
print m
m.train(40)
m.test()

# with 10 iterations
# Sentences: 7422
# Micro-averaged measures:
#   Precision: 60.5
#   Recall: 76.8
#   Harmonic mean F1: 67.6

# m2 = DMVCCM(tb)
# m2.train(10)
# m2.test()
