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


class CCM(model.BracketingModel):
    # ESTE NUMERO ES MUY IMPORTANTE: CON 2 SE CLAVA PA LA BOSTA,
    # CON 1 ANDA MUCHO MEJOR
    # ESTO ES PORQUE CON l=1 cazamos para ... A B C ... el contexto (A,C)
    # pongo 0 si quiero modelar empty spans:
    min_span_size = 0
    extra_constituents = 2.0
    extra_distituents = 8.0
    extra_counts = extra_constituents + extra_distituents
    
    # Used in the MStep:
    # if it is 1, START X END adds negative evidence for the contexts
    # (START, X) and (X, END).
    min_sentence_length = 2
    
    def __init__(self, treebank=None, training_corpus=None):
        model.BracketingModel.__init__(self, treebank, training_corpus)
        random.seed(0)
        self.tita = None
        self.count = None
    
    def init_count(self):
        count = paramdict.ParamDict()
        #for s in self.S:
        for s in self.training_corpus.tagged_sents():
            s = [x[1] for x in s]
            n = len(s)
            s = s + ['END', 'START']
            for l in range(self.min_span_size, n+1):
                for i in range(n-l+1):
                    j = i + l
                    alpha = string.join(s[i:j])
                    beta = (s[i-1], s[j])
                    other = self.other_features(s, i, j)
                    count.add1(alpha)
                    count.add1(beta)
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
        
        # No da igual si se consideran empty spans:
        for s in itertools.ifilter(lambda s: len(s) >= self.min_sentence_length, \
                                    self.training_corpus.tagged_sents()):
            s = [x[1] for x in s]
            s = s + ['END', 'START']
            p_bracket = self.p_bracket(s)
            self.MStep_s(s, p_bracket, tita)
        
        for (a, b), p in tita.iteritems():
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
                p_b = p_bracket[i, j]
                tita.add((alpha, True), p_b)
                tita.add((alpha, False), 1.0 - p_b)
                tita.add((beta, True), p_b)
                tita.add((beta, False), 1.0 - p_b)
                for o in other:
                    tita.add((o, True), p_b)
                    tita.add((o, False), 1.0 - p_b)
    
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
            log_prob_s = math.log(I[0, n])
            log_prob_s += math.log(1.0/bracketing.binary_bracketings_count(n))
            tita = self.tita
            for i in range(n):
                for j in range(i, n+1):
                    alpha = string.join(s[i:j])
                    beta = (s[i-1], s[j])
                    log_prob_s += math.log(tita.val((alpha, False)))+math.log(tita.val((beta, False)))
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
        
        return result
    
    def IO(self, s):
        """s must come with ['END', 'START']."""
        n = len(s) - 2
        
        I = self.I(s)
        
        O = {}
        O[0, n] = 1.0
        
        # l va de n-1 a 2.
        for l in range(n-1, 1, -1):
            for i in range(n-l+1):
                j = i + l
                sum1 = sum(I[k, i]*O[k, j] for k in range(i))
                sum2 = sum(I[j, k]*O[i, k] for k in range(j+1, n+1))
                O[i, j] = self.phi(i, j, s) * (sum1 + sum2)
        
        return (I, O)
    
    # Ver tesis de Klein, A.1:
    def I(self, s):
        """s must come with ['END', 'START']."""
        I = {}
        
        n = len(s) - 2
        for l in range(1, n+1):
            for i in range(n-l+1):
                j = i + l
                if l == 1:
                    I[i, j] = 1.0
                elif l == 2:
                    I[i, j] = self.phi(i, j, s)
                elif l == n:
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
