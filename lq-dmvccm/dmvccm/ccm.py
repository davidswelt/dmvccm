# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

#!/usr/bin/python
# -*- coding: iso-8859-1 -*-

# ccm.py: Implementation of Klein's CCM (generative version).
# 
# TODO: remove the unnecessary (*, False) parameters from CCM.


import string
import itertools
import random
import math

import bracketing
import util
import model
import paramdict
import sys,traceback

recency_feature = False

counter_threshold = 0.1
def inc_float_counter(counter,key):
    if key in counter:
        counter[key] += 1.0
    else:
        counter[key] = 1.0

def devalue_counter(counter):
    # de-value occurrences of previous sequences
    for key in counter.keys():
        counter[key] /= 2.0
        if counter[key] < counter_threshold:
            del counter[key]
        
def prune_counter(counter):
    for key in counter.keys():
        if counter[key] < counter_threshold:
            del counter[key]

def above_threshold(counter, key):
    return key in counter and counter[key]>counter_threshold

class CCM(model.BracketingModel):
    # ESTE NUMERO ES MUY IMPORTANTE: CON 2 SE CLAVA PA LA BOSTA,
    # CON 1 ANDA MUCHO MEJOR
    # ESTO ES PORQUE CON l=1 cazamos para ... A B C ... el contexto (A,C)
    # pongo 0 si quiero modelar empty spans:
    min_span_size = 0
    extra_constituents = 2.0
    extra_distituents = 8.0
    extra_counts = extra_constituents + extra_distituents

    # cache (for recency)
    prev_seqs = {}

    
    
    # Used in the MStep:
    # if it is 1, START X END adds negative evidence for the contexts
    # (START, X) and (X, END).
    min_sentence_length = 2


    # constants
    @staticmethod
    def previous(key):
        return "previous_"+key
    
    def __init__(self, treebank=None, training_corpus=None):
        model.BracketingModel.__init__(self, treebank, training_corpus)
        random.seed(0)
        self.tita = None
        self.count = None

        
        
    def init_count(self):
        count = paramdict.ParamDict()
        #for s in self.S:

        
        for s in self.training_corpus.tagged_sents():
            s = [x[1] for x in s]  # tags  [would be easy to add lexical component here]
            n = len(s)
            s = s + ['END', 'START']
            for l in range(self.min_span_size, n+1):
                for i in range(n-l+1):
                    j = i + l
                    alpha = string.join(s[i:j])  # yield
                    beta = (s[i-1], s[j])  # context
                    other = self.other_features(s, i, j) # further features
                    count.add1(alpha)
                    count.add1(beta)
                    if recency_feature:
                        count.add1(self.previous(alpha))
                    for o in other:
                        count.add1(o)

                        
            
        self.count = count
    
    def train(self, n=1):
        self.steps(n)
        self.eval()
    
    def steps(self, n):
        for i in range(n):
            self.step()
            self.test(short=True)
    
    def step(self):
        self.MStep()
        self.EStep()
        print 'Log-likelihood:', self.Log_likelihood    
        # self.tita.show(None, self.previous(""))
    
    def other_features(self, s, i, j):
        return []
    
    def MStep(self):
        extra_consts = self.extra_constituents
        extra_dists = self.extra_distituents
        extra_counts = self.extra_counts
        
        # the default value is needed only in the parser if it is used with
        # sentences that weren't in the training.
        tita = CCMTita(default_val=extra_consts/extra_counts)
        self.Log_likelihood = 0.0
        
        count = self.count
        if count is None:
            self.init_count()
            count = self.count

        self.prev_seqs = {}  # history

            
        # No da igual si se consideran empty spans:
        for s in itertools.ifilter(lambda s: len(s) >= self.min_sentence_length, \
                                    self.training_corpus.tagged_sents()):
            s = [x[1] for x in s]
            s = s + ['END', 'START']
            p_bracket = self.p_bracket(s)
            self.MStep_s(s, p_bracket, tita)

            devalue_counter(self.prev_seqs)

        print self.prev_seqs
        

        for (a, b), p in tita.iteritems():
            # extra_consts: smoothing
            if b:
                tita.setVal((a, b), (p+extra_consts) / (count.val(a)+extra_counts))
            else:
                tita.setVal((a, b), (p+extra_dists) / (count.val(a)+extra_counts))
        
        self.tita = tita
    
    def MStep_s(self, s, p_bracket, tita):
        """s must come with ['END', 'START']."""
        n = len(s) - 2
        #s = s + ['END', 'START']
        for l in range(self.min_span_size, n+1):
            for i in range(n-l+1):
                j = i + l
                alpha = string.join(s[i:j])
                beta = (s[i-1], s[j])
                other = self.other_features(s, i, j)
                p_b = p_bracket[i, j]  # DR: is this the "current theta"?

                if p_b<0 or p_b>1.0:
                    print "warning -- p_b=",p_b, i,j,alpha
                    # asd = 5/0

                    
                tita.add((alpha, True), p_b)
                tita.add((alpha, False), 1.0 - p_b)
                tita.add((beta, True), p_b)
                tita.add((beta, False), 1.0 - p_b)
                for o in other:
                    tita.add((o, True), p_b)
                    tita.add((o, False), 1.0 - p_b)

                # TODO:
                # why is alpha empty sometimes?    

                # alpha_active = self.previous(alpha)
                # tita.add((alpha_active, True), p_b)
                # tita.add((alpha_active, False), 1.0 - p_b)

                
                if recency_feature:
                    alpha_active = self.previous(alpha)
                    if len(alpha)>1 and above_threshold(self.prev_seqs,alpha):  # arbitrary threshold
                        # print "found previous", alpha,  self.prev_seqs[alpha]

                        tita.add((alpha_active, True), p_b)
                        tita.add((alpha_active, False), 1.0 - p_b)

                        # TODO - is the following correct?
                        # first intuition was not to add p_b at all if no recent occurrence was there
                        # maybe we need to add 1.0 instead of p_b?
                    else:                        
                        tita.add((alpha_active, True), 0.5)
                        tita.add((alpha_active, False), 0.5)

                    
                # note use of sequence
                inc_float_counter(self.prev_seqs, alpha) 

    
    def EStep(self):
        pass
    
    def p_bracket(self, s):
        """s must come with ['END', 'START']."""
        if self.tita is None:
            init = True
        else:
            init = False

        n = len(s) - 2
        
        result = {}
        
        # optimizacion:
        for i in range(n):
            result[i, i] = 0.0
            result[i, i+1] = 1.0
        result[n, n] = 0.0
        result[0, n] = 1.0
        
        if not init:
            (I, O) = self.IO(s)

            # compute log(P(s)) with P(s) = K(s) * I(0, n, s)
            # XXX: it is only correct for CCM2.
            log_prob_s = 0
            try:
                log_prob_s = math.log(I[0, n])
                log_prob_s += math.log(1.0/bracketing.binary_bracketings_count(n))
            except ValueError:
                print "ValueError with I[0,n]=",I[0,n]
                print "or with bracketing.binary_bracketings_count(n)=",bracketing.binary_bracketings_count(n)
                traceback.print_exc(file=sys.stdout)
                
            tita = self.tita
            for i in range(n):
                for j in range(i, n+1):
                    alpha = string.join(s[i:j])
                    beta = (s[i-1], s[j])

                    a_t = tita.val((alpha, False))
                    b_t = tita.val((beta, False))
                    p_t = tita.val((self.previous(alpha), False))
                    if a_t<0 or b_t<0 or p_t<0:
                        print "error: ",alpha, a_t,b_t,p_t
                    else:
                        log_prob_s += math.log(a_t)+math.log(b_t)
                        if recency_feature:
                            log_prob_s += math.log(p_t)
                    # TODO: What about "other" features?
                    
                    #if log_prob_s == 0.0:
                    #    print 'la caca es', alpha, beta

                    
            self.Log_likelihood += log_prob_s
            #self.Log_likelihood += math.log(I[0, n])

        # antes l llegaba hasta n pero lo optimice:
        for l in range(2, n):
            for i in range(n-l+1):
                j = i + l
                if init:
                    result[i, j] = p_bracket_split(i, j, n)
                    # result[i, j] = p_bracket_bin(i, j, n)
                else:
                    # XXX: para la subclase SimpleCCMPunct que a veces devuelve phi = 0. creo que sirve para otras subclases de *Punct.
                    if I[i, j]*O[i, j] == 0.0:
                        result[i, j] = 0.0
                    else:
                        result[i, j] = I[i, j]*O[i, j]/(I[0, n]*self.phi(i, j, s))
                        # why / phi?
                        # is this P_b(i,j) rather than P_b(i,j|s)?  but phi(i,j,s) is not p_b(s)
                        # maybe because I or O each contain phi?
                        
        return result
    
    def IO(self, s):
        """s must come with ['END', 'START']."""
        n = len(s) - 2
        
        I = self.I(s)
        O = {}
        O[0, n] = 1.0  # case j-i==m

        # thesis p. 104 
        
        # l va de n-1 a 2.
        for l in range(n-1, 1, -1):
            for i in range(n-l+1):
                j = i + l
                sum1 = sum(I[k, i]*O[k, j] for k in range(i))   # left context
                sum2 = sum(I[j, k]*O[i, k] for k in range(j+1, n+1))  # right context
                O[i, j] = self.phi(i, j, s) * (sum1 + sum2)
                # thesis specifies phi*sum1 + sum2, but that is likely incorrect anyway
                
        return (I, O)
    
    # Ver tesis de Klein, A.1:
    def I(self, s):
        """s must come with ['END', 'START']."""
        I = {}
        
        n = len(s) - 2
        for l in range(1, n+1):
            for i in range(n-l+1):
                j = i + l
                # j here is j-1 in Klein's thesis appendix
                if l == 1:   # case j-i==0
                    I[i, j] = 1.0
                elif l == 2: # case j-i==1
                    I[i, j] = self.phi(i, j, s)
                elif l == n: # not listed (optimization? phi==1?)
                    I[i, j] = sum(I[i, k] * I[k, j] for k in range(i+1, j))
                else:
                    I[i, j] = self.phi(i, j, s) * sum(I[i, k] * I[k, j] for k in range(i+1, j))
        
        return I
    
    def parse(self, s):
        parse = {}
        
        n = len(s)
        s = s + ['END', 'START']
        for l in range(1, n+1):
            for i in range(n-l+1):
                j = i + l
                if l == 1:
                    parse[i, j] = (1.0, [])
                elif l == 2:
                    parse[i, j] = (self.phi(i, j, s), [(i, j)])
                else:
                    max, k_max = None, []
                    for k in range(i+1, j):
                        x = parse[i, k][0] * parse[k, j][0]
                        if (max is None) or (max < x):
                            k_max, max = [k], x
                        elif max == x:
                            k_max = k_max + [k]
                    k_max = random.choice(k_max)
                    # elegir el primer elemento tiende mas a rbranch,
                    # sin embargo da (casi) el mismo resultado: 68.9
                    #k_max = k_max[0]
                    
                    parse[i, j] = (max * self.phi(i, j, s), [(i, j)] + parse[i, k_max][1] + parse[k_max, j][1])
        
        result = (bracketing.Bracketing(n, set(parse[0, n][1][1:])), parse[0, n][0])

        devalue_counter(self.prev_seqs)

        return result
    
    def phi(self, i, j, s):
        """s must come with ['END', 'START']."""
        if self.tita is None:
            return 1.0
        n = len(s) - 2
        #s = s + ['END', 'START']
        tita = self.tita
        alpha = string.join(s[i:j])
        beta = (s[i-1], s[j])
        # j -i == 1 y j - i == len(s) podrian estar fuera del dominio de phi tambien.
        if j - i == 1:
            # assert False
            result = tita.val((beta, True))
        elif j - i == n:
            # se invoca en parse() y en IO()/I() creo que tambien
            # assert False
            result = tita.val((alpha, True))
        else:

            if tita.val((alpha, False)) * tita.val((beta, False)) == 0:
                print 'i j s', i, j, s
                print 'alpha beta', alpha, beta
                print 'tita[alpha, False] tita[beta, False]', tita.val((alpha, False)), tita.val((beta, False))
            result = (tita.val((alpha, True)) * tita.val((beta, True))) / (tita.val((alpha, False)) * tita.val((beta, False)))
            # print "phi-result %s,%s:%s"%(i,j,result)


                        
            # recency
            if recency_feature:
                if above_threshold(self.prev_seqs,alpha):
                    # recency feature
                    #if tita.val((self.previous(alpha), False)) != 0:
                    previous_alpha = self.previous(alpha)
                    # if tita.d.has_key((previous_alpha, False)):
                    if tita.val((previous_alpha, True)) / tita.val((previous_alpha, False)) > 1.0:
                        print result, tita.val((previous_alpha, True)) , tita.val((previous_alpha, False))
                    result *= tita.val((previous_alpha, True)) / tita.val((previous_alpha, False))
                    # print "-->", result
                    if result<0:
                        print "result<0! ", result
                
                inc_float_counter(self.prev_seqs, alpha)
            
            other = self.other_features(s, i, j)
            for o in other:
                if tita.d.has_key((o, True)) and tita.d.has_key((o, False)):
                    result *= tita.val((o, True)) / tita.val((o, False))

                    
        return result


def p_bracket_split(i, j, n):
    if i == j:
        res = 0.0
    elif i == 0 and j == n:
        res = 1.0
    elif (0 < i) and (i < j) and (j < n):
        res = 2.0 / ((j-i)*(j-i+1))
    else:
        res = 1.0 / (j-i)
    return res


# Tablitas boludas para calcular p_bracket_bin:
fac = [1.0]
for i in range(1,40):
    fac += [fac[i-1]*i]

def comb(n, m):
    return fac[n]/(fac[n-m]*fac[m])

T = [1.0]
for i in range(1, 12):
    T += [comb(2*i-2, i-1)/i]

def p_bracket_bin(i, j, n):
    if i == j:
        res = 0.0
    else:
        res = (T[j-i] * T[n-(j-i-1)]) / T[n]
    return res


class CCMTita(paramdict.ParamDict):
    """Keys are pairs (a, b) with b boolean. If b, the default value is
    self.default_val. If not b, it is 1.0 - self.default_val.
    """
   
    def val(self, x):
        if x[1]:
            return self.d.get(x, self.default_val)
        else:
            return self.d.get(x, 1.0 - self.default_val)


    def show(self,maxlen=None,prefix=None):
        count = 0
        for k,v in self.d.iteritems():
            a,b = k
            if not maxlen or len(a)<=maxlen:
                if not prefix or (isinstance(a,basestring) and a.startswith(prefix)):
                    print a,b,v
                    count += 1
        fs = ""
        if maxlen:
            fs += " of length %s or less"%maxlen
        if prefix:
            fs += " with prefix %s"%prefix
        print "%s items%s found."%(count,fs)








        
# CCM2: This is the really generative version. CCM is discriminative, but it can be
# proved that they are equivalent.
class CCM2(CCM):
    
    def MStep(self):
        tita = CCMTita()
        self.Log_likelihood = 0.0
        p_true = 0.0
        p_false = 0.0
        for s in itertools.ifilter(lambda s: len(s) > 1, \
                                    self.training_corpus.tagged_sents()):
            s = [x[1] for x in s]
            n = len(s)
            s = s + ['END', 'START']
            p_bracket = self.p_bracket(s)
            
            for l in range(self.min_span_size, n+1):
                for i in range(n-l+1):
                    j = i + l
                    alpha = string.join(s[i:j])
                    beta = (s[i-1], s[j])
                    other = self.other_features(s, i, j)
                    if self.tita is None:
                        p_b = p_bracket_split(i, j, n)
                        #p_b = p_bracket_bin(i, j, n)
                    else:
                        p_b = p_bracket[i, j]
                    tita.add((alpha, True), p_b)
                    tita.add((alpha, False), 1.0 - p_b)
                    tita.add((beta, True), p_b)
                    tita.add((beta, False), 1.0 - p_b)
                    p_true += p_b
                    p_false += 1.0 - p_b
                    for o in other:
                        tita.add((o, True), p_b)
                        tita.add((o, False), 1.0 - p_b)
        
        # smoothing
        for (a, b), p in tita.iteritems():
            if b:
                tita.add((a, b), self.extra_constituents)
                p_true += self.extra_constituents
            else:
                tita.add((a, b), self.extra_distituents)
                p_false += self.extra_distituents
        
        for (a, b), p in tita.iteritems():
            if b:
                #tita.setVal((a, b), (p+self.extra_constituents) / (p_true+self.extra_counts))
                tita.setVal((a, b), p / p_true)
            else:
                #tita.setVal((a, b), (p+self.extra_distituents) / (p_false+self.extra_counts))
                tita.setVal((a, b), p / p_false)
        
        self.tita = tita


def test():
    m = CCM()
    return m

def test2():
    import negra10
    tb = negra10.Negra10()
    m = CCM(tb)
    return m

def test3(simplify=False):
    import cast3lb10
    tb = cast3lb10.Cast3LB10()
    if simplify:
        tb.simplify_tags()
    m = CCM(tb)
    return m

def test4():
    import cast3lb10
    tb = cast3lb10.Cast3LB10()
    tb.simplify_tags()
    t = tb.trees*10
    tb.trees = t

    m = CCM(tb)
    return m
