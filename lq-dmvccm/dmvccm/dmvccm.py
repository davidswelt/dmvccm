# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# dmvccm.py: DMV+CCM with one-side-first constraint.

import itertools
import time
import math
import random

from nltk import tree

# FIXME: use absolute package path!
import dmv
import ccm


class DMVCCM(dmv.DMV):
    # The same symbol as CCM's:
    end_symbol = 'END'
    only_dmv = False
    # Inherited parameter:
    # use_adj = True
    
    # set to True if you want to make p_order = 0.5 always.
    constant_p_order = False
    
    def __init__(self, treebank=None, only_dmv=False):
        self.only_dmv = only_dmv
        dmv.DMV.__init__(self, treebank=treebank)
        if not only_dmv:
            self.ccm = ccm.CCM(treebank)
        self.tita = None
    
    def MStep(self):
        tita, count = ccm.CCMTita(count_evidence=True), ccm.CCMTita()
        ccm_tita = ccm.CCMTita()
        self.Log_likelihood = 0.0
        
        # Da igual de las dos maneras?
        #for s in self.S:
        #for s in itertools.ifilter(lambda s: len(s) > 1, self.S):
        for s in itertools.ifilter(lambda s: len(s) > 1, \
                                    self.training_corpus.tagged_sents()):
            s = [x[1] for x in s]
            s2 = s + [self.end_symbol]
            if self.only_dmv or self.ccm.tita is not None:
                pio = self.p_inside_outside(s2)
                self.Log_likelihood += math.log(pio.denom)
                self.MStep_s(s2, pio, tita, count)
            if not self.only_dmv:
                if self.ccm.tita is not None:
                    p_bracket = self.p_bracket(s, pio)
                else:
                    # En el primer paso usamos p_bracket viejo que usa p_split:
                    p_bracket = self.ccm.p_bracket(s + ['END', 'START'])
                self.ccm.MStep_s(s + ['END', 'START'], p_bracket, ccm_tita)

        for x, p in tita.iteritems():
            if p == 0.0:
                pass
            elif not self.constant_p_order and x[0] == 'order_left':
                # x[1] == h
                tita.setVal(x, p / count.val(x[1]))
            elif x[0] == 'stop_left':
                # x[1] == h, x[2] == adj
                tita.setVal(x, p / count.val(x))
            elif x[0] == 'stop_right':
                # x[1] == h, x[2] == adj
                tita.setVal(x, p / count.val(x))
            elif x[0] == 'attach_left':
                # x[1] == a, x[2] == h
                if count.val(('attach_left', x[2])) == 0.0:
                    # self.count = count
                    # self.new_tita = tita
                    print x, p
                tita.setVal(x, p / count.val(('attach_left', x[2])))
            elif x[0] == 'attach_right':
                # x[1] == a, x[2] == h
                tita.setVal(x, p / count.val(('attach_right', x[2])))
            p_new = tita.val(x)
            if not (-0.01 <= p_new and p_new <= 1.01):
            #if p_new <= -0.01:
                self.count = count
                #self.new_tita = tita
                print '(x, p, p_new) =', (x, p, p_new)
            # assert 0.0 <= p_new and p_new <= 1.0
        
        if self.only_dmv or self.ccm.tita is not None:
            self.tita = tita
        
        if not self.only_dmv:
            extra_consts = self.ccm.extra_constituents
            extra_dists = self.ccm.extra_distituents
            extra_counts = self.ccm.extra_counts

            ccm_count = self.ccm.count
            if ccm_count is None:
                self.ccm.init_count()
                ccm_count = self.ccm.count
            
            # Para reestimar CCM:
            for (a, b), p in ccm_tita.iteritems():
                if b:
                    ccm_tita.setVal((a, b), (p+extra_consts) / (ccm_count.val(a)+extra_counts))
                else:
                    ccm_tita.setVal((a, b), (p+extra_dists) / (ccm_count.val(a)+extra_counts))
            self.ccm.tita = ccm_tita
    
    # s comes without END at the end.
    def p_bracket(self, s, pio):
        # assert self.ccm.tita is not None
        s2 = s + ['END']
        n = len(s)
        n2 = n + 1
        (pi, po) = (pio.pi, pio.po)
        end_seal = dmv.Node('|', self.end_symbol, n2-1)
        
        def p_head(k, i, j):
            return self.p_head(k, i, j, s2, pio, end_seal, n2)
        
        result = {}
        
        # optimization:
        for i in range(n):
            result[i, i] = 0.0
            result[i, i+1] = 1.0
        result[n, n] = 0.0
        result[0, n] = 1.0
        
        # antes l llegaba hasta n pero lo optimice:
        for l in range(2, n):
            for i in range(n-l+1):
                j = i + l
                result[i, j] = 0.0
                for k in range(i, j):
                    result[i, j] += p_head(k, i, j)
                
                if not ((-0.01 <= result[i, j]) and (result[i, j] <= 1.0001)):
                    print 'Se pasoo result[i ,j]', result[i, j]
                    print 's, i, j', s, i, j
                assert (-0.01 <= result[i, j]) and (result[i, j] <= 1.0001)
        
        return result
    
    # Invoked by p_bracket. It is apart so it can be debugged.
    # s2 comes with END at the end.
    def p_head(self, k, i, j, s2, pio, end_seal, n2):
        (pi, po) = (pio.pi, pio.po)
        h = s2[k]
        hk = h+str(k)
        denom = pi[0, n2].val(end_seal) * self.phi(i, j, s2)
        r1 = (pi[i, j].val('<>'+hk) - pi[i, j].val('>'+hk)*self.p_stop_right(h, k==j-1)) * po[i, j].val('<>'+hk) / denom
        r2 = (pi[i, j].val('><'+hk) - pi[i, j].val('<'+hk)*self.p_stop_left(h, k==i)) * po[i, j].val('><'+hk) / denom
        return r1 + r2 + self.c_s(pio, s2, '<'+hk, i, j) + self.c_s(pio, s2, '>'+hk, i, j)
    
    # s comes with END at the end.
    def c_s(self, pio, s, node, i, j):
        if j == len(s):
            return 0.0
        # phi puede ser 0 en las subclases
        phi = self.phi(i, j, s)
        if phi == 0.0:
            return 0.0
        else:
            return dmv.DMV.c_s(self, pio, s, node, i, j) / phi
    
    # FIXME: too much repeated code only to add the phi factor. 
    # better add i,j parameters to p_attach_*.
    # s comes with END at the end.
    def p_inside_outside(self, s):
        p_inside = self.p_inside(s)
        
        n = len(s)
        p_outside = {}
        # OPTIMIZATION: code the result directly
        #base_list = []
        #node = dmv.Node('|', self.end_symbol, n)
        #base_list = self.unary_p_outside(1.0, node, 0, n2)
        #p_outside[0, n] = dmv.PIODict(base_list)
        p_outside[0, n] = dmv.PIODict([(1.0, dmv.Node('|', self.end_symbol, n-1)), (1.0, dmv.Node('<>', self.end_symbol, n-1))])
        
        # OPTIMIZATION: END considered only explicitly
        n = n-1
        
        # OPTIMIZATION: separately compute the first step
        base_list = []
        phi = self.phi(0, n, s)
        for i in range(n):
            w = s[i]
            prob = self.p_attach_left(w, self.end_symbol, n-i) * phi
            node = dmv.Node('|', w, i)
            base_list += self.unary_p_outside(prob, node, 0, n)
        p_outside[0, n] = dmv.PIODict(base_list)
        
        # l goes from n-1 to 1.
        for l in range(n-1, 0, -1):
            for i in range(n-l+1):
                j = i + l
                phi = self.phi(i, j, s)
                
                p_outside_dict = dmv.PIODict()
                
                for k in range(i):
                    for (p1, n1) in p_inside[k, i].itervalues():
                        for (p2, n2) in p_outside[k, j].itervalues():
                            if n1 == n2 and n1.mark[0] == '>':
                                h = n1.word
                                for m in range(i, j):
                                    a = s[m]
                                    # m-n1.index = distancia entre uno y otro
                                    # DMVCCM: multiplicar por phi:
                                    p3 = self.p_nonstop_right(h, n1.index==i-1) * \
                                        self.p_attach_right(a, h, m-n1.index) * \
                                        p1 * p2 * phi
                                    n3 = dmv.Node('|', a, m)
                                    p_outside_dict.add(p3, n3)
                            elif n1.mark == '|' and n2.mark[0] == '<' and i <= n2.index and n2.index < j:
                                # m-n1.index = distancia entre uno y otro
                                m = n2.index
                                h = n2.word
                                # DMVCCM: multiplicar por phi:
                                p3 = self.p_nonstop_left(h, m==i) * \
                                    self.p_attach_left(n1.word, h, m-n1.index) * \
                                    p1 * p2 * phi
                                n3 = dmv.Node(n2.mark, h, m)
                                p_outside_dict.add(p3, n3)
                for k in range(j+1, n+1):
                    for (p1, n1) in p_inside[j, k].itervalues():
                        for (p2, n2) in p_outside[i, k].itervalues():
                            if n1 == n2 and n1.mark[0] == '<':
                                h = n1.word
                                for m in range(i, j):
                                    a = s[m]
                                    # n1.index-m = distancia entre uno y otro
                                    # DMVCCM: multiplicar por phi:
                                    p3 = self.p_nonstop_left(h, n1.index==j) * \
                                        self.p_attach_left(a, h, n1.index-m) * \
                                        p1 * p2 * phi
                                    n3 = dmv.Node('|', a, m)
                                    p_outside_dict.add(p3, n3)
                            elif n1.mark == '|' and n2.mark[0] == '>' and i <= n2.index and n2.index < j:
                                m = n2.index
                                h = n2.word
                                # n1.index-m = distancia entre uno y otro
                                # DMVCCM: multiplicar por phi:
                                p3 = self.p_nonstop_right(h, m==j-1) * \
                                    self.p_attach_right(n1.word, h, n1.index-m) * \
                                    p1 * p2 * phi
                                n3 = dmv.Node(n2.mark, h, m)
                                p_outside_dict.add(p3, n3)
                
                if l == 1:
                    end = True
                else:
                    end = False
                p_outside[i, j] = dmv.PIODict(sum((self.unary_p_outside(p, node, i, j, end) for (p, node) in p_outside_dict.itervalues()), []))
        
        # XXX: aca faltaria agregar p_outside[n, n+1]
        
        return dmv.PIO(n+1, p_inside, p_outside, self.end_symbol)
    
    # FIXME: too much repeated code only to add the phi factor. 
    # better add i,j parameters to p_attach_*.
    # s comes with END at the end.
    def p_inside(self, s):
        p_inside = {}
        # OPTIMIZATION: esta sutileza hace casi todo:
        #n = len(s)
        n = len(s) - 1
        
        for i in range(n):
            j = i + 1
            w = s[i]
            # DMVCCM: multiply by phi: I am assuming every I[i,j] has the
            # factor phi. Maybe this assumption can be modified.
            phi = self.phi(i, j, s)
            #phi = 1.0
            pl = self.p_order('left', w) * phi
            pr = self.p_order('right', w) * phi
            n0 = dmv.Node('<', w, i)
            n1 = dmv.Node('>', w, i)
            p_inside[i, j] = dmv.PIODict(self.unary_p_inside(pl, n0, i, j) + self.unary_p_inside(pr, n1, i, j))
        
        for l in range(2, n+1):
            for i in range(n-l+1):
                j = i + l
                # tenemos p_inside[a, b] para todas las cosas adentro de (i, j).
                p_inside_dict = dmv.PIODict()
                phi = self.phi(i, j, s)
                for k in range(i+1, j):
                    for (p1, n1) in p_inside[i, k].itervalues():
                        for (p2, n2) in p_inside[k, j].itervalues():
                            if n1.mark[0] == '>' and n2.mark == '|':
                                m = n1.index
                                h = n1.word
                                # n2.index-m = distancia entre uno y otro
                                # DMVCCM: multiplicar por phi:
                                p = self.p_nonstop_right(h, m==k-1) * \
                                    self.p_attach_right(n2.word, h, n2.index-m) * \
                                    p1 * p2 * phi
                                p_inside_dict.add(p, n1)
                            if n1.mark == '|' and n2.mark[0] == '<':
                                m = n2.index
                                h = n2.word
                                # m-n1.index = distancia entre uno y otro
                                # DMVCCM: multiplicar por phi:
                                p = self.p_nonstop_left(h, m==k) * \
                                    self.p_attach_left(n1.word, h, m-n1.index) * \
                                    p1 * p2 * phi
                                p_inside_dict.add(p, n2)
                
                # here is where the stops are generated:
                p_inside[i, j] = dmv.PIODict(sum((self.unary_p_inside(p, node, i, j) for (p, node) in p_inside_dict.itervalues()), []))
        
        # OPTIMIZATION: finally, choose the head of the sentence.
        p_sum = 0.0
        for i in range(n):
            w = s[i]
            # DMVCCM: the phi factor comes with p1.
            p1 = p_inside[0, n].val('|'+w+str(i))
            p_sum += p1 * self.p_attach_left(w, self.end_symbol, n-i)
        p_inside_dict = dmv.PIODict()
        #p_inside_dict.add(p_sum, dmv.Node('<', self.end_symbol, n))
        p_inside_dict.add(p_sum, dmv.Node('|', self.end_symbol, n))
        p_inside[0, n+1] = p_inside_dict
        
        return p_inside
    
    # FIXME: too much repeated code only to add the phi factor. 
    # better add i,j parameters to p_attach_*.
    # s comes without END at the end.
    def dep_parse(self, s):
        parse = {}
        # OPTIMIZATION: END considered only explicitly
        # s = s + [self.end_symbol]
        # solo para uso como param. de phi
        s2 = s + [self.end_symbol]
        n = len(s)
        
        for i in range(n):
            j = i + 1
            # >w -> w
            # <w -> w
            w = s[i]
            # DMVCCM: multiplicar por phi:
            # aca da lo mismo:
            #phi = self.phi(i, j, s)
            phi = 1.0
            pl = self.p_order('left', w) * phi
            pr = self.p_order('right', w) * phi
            t0 = tree.Tree(dmv.Node('<', w, i), [w])
            t1 = tree.Tree(dmv.Node('>', w, i), [w])
            
            parse[i, j] = dmv.ParseDict(self.unary_parses(pl, t0, i, j) + self.unary_parses(pr, t1, i, j))
        
        for l in range(2, n+1):
            for i in range(n-l+1):
                j = i + l
                # tenemos parse[a, b] para todas las cosas adentro de (i, j).
                parse_dict = dmv.ParseDict()
                phi = self.phi(i, j, s2)
                for k in range(i+1, j):
                    # aqui, mejores parses entre parse[i, k] y parse[k, j]
                    for (p1, t1) in parse[i, k].itervalues():
                        for (p2, t2) in parse[k, j].itervalues():
                            n1 = t1.node
                            n2 = t2.node
                            if n1.mark[0] == '>' and n2.mark == '|':
                                m = n1.index
                                h = n1.word
                                # n2.index-m = distancia entre uno y otro
                                # DMVCCM: multiplicar por phi:
                                p = self.p_nonstop_right(h, m==k-1) * \
                                    self.p_attach_right(n2.word, h, n2.index-m) * \
                                    p1 * p2 * phi
                                t = tree.Tree(n1, [t1, t2])
                                parse_dict.add(p, t)
                            if n1.mark == '|' and n2.mark[0] == '<':
                                m = n2.index
                                h = n2.word
                                # m-n1.index = distancia entre uno y otro
                                # DMVCCM: multiplicar por phi:
                                p = self.p_nonstop_left(h, m==k) * \
                                    self.p_attach_left(n1.word, h, m-n1.index) * \
                                    p1 * p2 * phi
                                t = tree.Tree(n2, [t1, t2])
                                parse_dict.add(p, t)
                
                # here is where the stops are generated:
                parse[i, j] = dmv.ParseDict(sum((self.unary_parses(p, t, i, j) \
                                    for (p, t) in parse_dict.itervalues()), []))
        
        # OPTIMIZATION: finally, choose the head of the sentence.
        #t_max, p_max = None, 0.0
        w = s[0]
        (p1, t1) = parse[0, n].val('|'+w+'0')
        t_max, p_max = t1, p1 * self.p_attach_left(w, self.end_symbol, n)
        # unbiased:
        l = [(t_max, p_max)]
        for i in range(1, n):
            w = s[i]
            (p1, t1) = parse[0, n].val('|'+w+str(i))
            p = p1 * self.p_attach_left(w, self.end_symbol, n-i)
            # aca hay bias (> seria elegir el primer head en caso de empate):
            # al parecer este bias solo afecta al modelo si no esta entrenado.
            # bias a RBRANCH:
            #if p > p_max:
            #    t_max, p_max = t1, p
            # bias a LBRANCH:
            #if p >= p_max:
            #    t_max, p_max = t1, p
            # unbiased:
            if p > p_max:
                p_max = p
                l = [(t1, p)]
            elif p == p_max:
                l += [(t1, p)]
        (t_max, p_max) = random.choice(l)
        
        return (t_max, p_max)
    
    # s va con el END puesto
    def phi(self, i, j, s):
        assert j != len(s)
        if self.only_dmv or j == len(s):
            return 1.0
        else:
            return self.ccm.phi(i, j, s + ['START'])
    
    def p_order(self, dir, w):
        # assert w != self.end_symbol
        if self.tita is None or self.constant_p_order:
            return 0.5
        p = self.tita.val(('order_left', w))
        if dir == 'left':
            return p
        else:
            return 1.0 - p


def test0(ntrees=10, steps=1, tb=None, clazz=None):
    if tb is None:
        import wsj10
        tb = wsj10.WSJ10()
    if clazz is None:
        clazz = DMVCCM
    tb.trees = tb.trees[:ntrees]
    m = clazz(tb)
    t0 = time.clock()
    m.train(steps)
    t = time.clock() - t0
    print 'Tiempo (seg.):', t
    return m


def test1(s=None):
    """s must be None or a string"""
    if s is None:
        s = "inside and outside"
    import treebank
    tb = treebank.Treebank([])
    m = DMVCCM(tb)
    return dmv.test_p_inside_outside(m, s.split())
