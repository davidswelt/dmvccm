lq-dmvccm - Franco M. Luque's Implementation of the DMV+CCM parser

Copyright (C) 2007-2011 Franco M. Luque
URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
For license information, see LICENSE.txt


Introduction
============

This package includes implementations of the CCM, DMV and DMV+CCM parsers from
Klein and Manning (2004), and code for testing them with the WSJ, Negra and 
Cast3LB corpuses (English, German and Spanish respectively). A detailed
description of the parsers can be found in Klein (2005).

This work was done as part of the PhD in Computer Science I am doing in FaMAF,
Universidad Nacional de Cordoba, Argentina, under the supervision of
Gabriel Infante-Lopez, with a research fellowship from CONICET.

Dependencies
============

    * ntlk: http://nltk.sourceforge.net
    * lq-nlp-commons: http://www.cs.famaf.unc.edu.ar/~francolq.


Installation and Configuration
==============================

Extract lq-nlp-commons.tar.gz and lq-dmvccm.tar.gz into a folder and add the
new folders to the Python PATH. For instance:

    # tar -zxvf lq-nlp-commons.tar.gz
    # tar -zxvf lq-dmvccm.tar.gz
    # export PYTHONPATH=`pwd`/lq-nlp-commons-0.1.0/:`pwd`/lq-dmvccm-0.1.0/


Usage
=====


Quickstart
----------

To train (with 10 iterations) and test the CCM parser with the WSJ10 do the 
following (replace 'my_wsj_path' with the path to the WSJ corpus):

    # python
    >>> import wsj10
    >>> tb = wsj10.WSJ10(basedir='my_wsj_path')
    >>> from dmvccm import ccm
    >>> m = ccm.CCM(tb)
    >>> m.train(10)
    >>> m.test()

To parse a sentence with the resulting parser do something like this:

    >>> s = 'DT NNP NN VBD DT VBZ DT JJ NN'.split()
    >>> (b, p) = m.parse(s)
    >>> b.brackets
    set([(6, 9), (1, 3), (4, 9), (4, 6), (3, 9), (0, 3), (7, 9)])
    >>> t = b.treefy(s)
    >>> t.draw()    


Parser Instantiation, Training and Testing
------------------------------------------

The WSJ10 corpus is used by default to train the parsers. The WSJ10 corpus is
automatically extracted from the WSJ corpus. You must place the WSJ corpus in a
folder named wsj_comb (or edit lq-nlp-commons/wsj.py, or read below).

For instance, to train (with 10 iterations) and test the CCM parser with the
WSJ10 do the following:

    # python
    >>> from dmvccm import ccm
    >>> m = ccm.CCM()
    >>> m.train(10)
    >>> m.test()

To give the treebanks explicitly:

    >>> import wsj10, negra10, cast3lb10
    >>> tb1 = wsj10.WSJ10()
    >>> tb2 = negra10.Negra10()
    >>> tb3 = cast3lb10.Cast3LB10()
    >>> m1 = ccm.CCM(tb1)
    >>> m2 = ccm.CCM(tb2)
    >>> m3 = ccm.CCM(tb3)

The Negra corpus must be in a file negra-corpus/negra-corpus2.penn (in Penn
format), and the Cast3LB corpus in a folder named 3lb-cast.

To use alternative locations for the treebanks use the parameter basedir when
creating the object. For instance:

    >>> import wsj10
    >>> tb = wsj10.WSJ10(basedir='my_wsj_path')

(similarly for Negra and Cast3LB)

When loaded for first time, the extracted corpuses are saved into files to avoid
having to process the entire treebanks again. The files are saved in the
working directory and must be there to be loaded in future instantiations of
the treebanks.

The DMV and DMV+CCM parsers are in the classes dmv.DMV and dmvccm.DMVCCM. The
current implementation of DMV+CCM has the one-side-first constraint. They are
used the same way CCM is. For instance:

    >>> from dmvccm import dmv, dmvccm
    >>> m1 = dmv.DMV()
    >>> m1.train(10)
    >>> m1.test()
    >>> m2 = dmvccm.DMVCCM()
    >>> m2.train(10)
    >>> m2.test()

Take into account that training DMV and DMV+CCM is much slower than training 
CCM. A single training step can take more than 20 minutes.


Parser Usage
------------

Once you have a trained instance of CCM, DMV or DMV+CCM, you can parse
sentences with the parse() method. You may give a list of words or an instance
of the sentence.Sentence class from lq-nlp-commons. The parse() method returns a
pair (b, p), where b is an instance of bracketing.Bracketing and p is the
probability of the bracketing. You can get the set of brackets from b.brackets
or convert the bracketing to a tree with b.treefy().

For instance:

    >>> from dmvccm import ccm
    >>> m = ccm.CCM()
    >>> m.train(2)
    >>> s = 'DT NNP NN VBD DT VBZ DT JJ NN'.split()
    >>> (b, p) = m.parse(s)
    >>> b.brackets
    set([(6, 9), (1, 3), (4, 9), (4, 6), (3, 9), (0, 3), (7, 9)])
    >>> t = b.treefy(s)
    >>> t.draw()

In the case of DMV and DMV+CCM you can also use the method dep_parse() to get
the dependency structure parsed by these models (Klein, 2005).

For instance:

    >>> from dmvccm import dmv
    >>> m = dmv.DMV()
    >>> m.train(2)
    >>> s = 'DT NNP NN VBD DT VBZ DT JJ NN'.split()
    >>> (t, p) = m.dep_parse(s)
    >>> t.draw()


References
==========

Klein, D. and Manning, C. D. (2004). Corpus-based induction of syntactic
structure: Models of dependency and constituency. In ACL, pages 478-485.

Klein, D. (2005). The Unsupervised Learning of Natural Language Structure. PhD
thesis, Stanford University.
