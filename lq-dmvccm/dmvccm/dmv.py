# Copyright (C) 2007-2011 Franco M. Luque
# URL: <http://www.cs.famaf.unc.edu.ar/~francolq/>
# For license information, see LICENSE.txt

# dmv.py: DMV with one-side-first constraint (class DMV), and without it 
# (class DMV2, not working yet). The initialization is made from initial
# values for the parameters (not as Klein does).

import itertools
import math
import random

from nltk import tree

import paramdict
import treebank
from dep import model
from dep import depset


class DMV(model.ProjDepModel):
    end_symbol = 'END'
    use_adj = True
    # if False, p_order is 0.5. if True, p_order is 1.0 for right-first.
    right_first = False
    
    # OLD! p_order reestimates to 0.5 always! moved to dmvccm.DMVCCM.
    constant_p_order = False
    
    def __init__(self, treebank=None, training_corpus=None):
        #model.BracketingModel.__init__(self, treebank=treebank)
        model.ProjDepModel.__init__(self, treebank, training_corpus)
        self.tita = None

    @staticmethod
    def tree_to_depset(t):
        res = set([(t.node.index, -1)])
        res.update(DMV._tree_to_depset(t))
        return depset.DepSet(len(t.leaves()), sorted(res))
    
    @staticmethod
    def _tree_to_depset(t):
        node = t.node
        index = node.index
        mark = node.mark
        #res = set([(index, -1)])
        if len(t) > 1:
            if mark[0] == '<':
                st = t[0]
            elif mark[0] == '>':
                st = t[1]
            res = set([(st.node.index, index)])
            res.update(DMV._tree_to_depset(t[0]), DMV._tree_to_depset(t[1]))
        else:
            if not isinstance(t[0], str):
                res = DMV._tree_to_depset(t[0])
            else:
                res = set()
        return res
    
    def train(self, n=1):
        # self.MStep()
        self.test(short=True)
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
        
    def EStep(self):
        pass
    
    def MStep(self):
        # XXX: esto reestima las cosas que tienen END tambien...
        # XXX: define new class DMVTita or use paramdict?
        #tita, count = paramdict.ParamDict(), paramdict.ParamDict()
        tita, count = DMVDict(), DMVDict()
        self.Log_likelihood = 0.0
        
        # Da igual de las dos maneras?
        #for s in self.S:
        #for s in itertools.ifilter(lambda s: len(s) > 1, self.S):
        for s in itertools.ifilter(lambda s: len(s) > 1, \
                                    self.training_corpus.tagged_sents()):
            s = [x[1] for x in s]
            s = s + [self.end_symbol]
            pio = self.p_inside_outside(s)
            self.Log_likelihood += math.log(pio.denom)
            self.MStep_s(s, pio, tita, count)
        
        for x, p in tita.iteritems():
            if p == 0.0:
                pass
            # XXX: p_order reestimates to 0.5 always!
            elif x[0] == 'order_left':
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
                """if count.val(('attach_left', x[2])) == 0.0:
                    self.count = count
                    self.new_tita = tita
                    print x, p"""
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
        
        self.tita = tita
    
    # LAS SIGUIENTES 4 FUNCIONES SE USAN EN MStep Y SIRVEN PARA SER
    # REDEFINIDAS EN LAS SUBCLASES:
    
    def MStep_s(self, s, pio, tita, count):
        """s must have 'END' at the end."""
        def c_s(node, i, j):
            return self.c_s(pio, s, node, i, j)
        
        n = len(s)
        (pi, po) = (pio.pi, pio.po)
        
        # Al END lo reestimamos aparte y tuti contenti...
        h = self.end_symbol
        den_attach_left = 1.0
        count.add(('attach_left', h), den_attach_left)
        for loca in range(n-1):
            a = s[loca]
            num_attach_left = c_s(Node('|', a, loca), 0, n-1)
            tita.add(('attach_left', a, h), num_attach_left)
        
        for loch in range(n-1):
            h = s[loch]
            h_seal = Node('|', h, loch)
            h_l = Node('<', h, loch)
            h_lr = Node('<>', h, loch)
            h_r = Node('>', h, loch)
            h_rl = Node('><', h, loch)
            
            # FIXME: code not used in this class but in subclass dmvccm.DMVCCM:
            # (DMV reestimates p_order always as 0.5 but DMVCCM doesn't).
            num_order_left = pi[loch, loch+1].val(h_l)
            tita.add(('order_left', h), num_order_left)
            count.add1(h)
            
            self.MStep_s_stop(pio, s, loch, tita, count)
            
            den_attach_left = 0.0
            for loca in range(loch):
                a = s[loca]
                num_attach_left = self.num_attach_left(pio, s, loch, loca)
                tita.add(('attach_left', a, h), num_attach_left)
                den_attach_left += num_attach_left
            count.add(('attach_left', h), den_attach_left)
            
            den_attach_right = 0.0
            for loca in range(loch+1, n-1):
                a = s[loca]
                num_attach_right = self.num_attach_right(pio, s, loch, loca)
                tita.add(('attach_right', a, h), num_attach_right)
                den_attach_right += num_attach_right
            count.add(('attach_right', h), den_attach_right)
    
    # s aparece por si las subclases quieren hace que incida en c_s.
    def c_s(self, pio, s, node, i, j):
        """s must have 'END' at the end."""
        # XXX: maybe c_s should never be invoked when j == len(s).
        if j == len(s):
            return 0.0
        else:
            return pio.c_s(node, i, j)
    
    def MStep_s_stop(self, pio, s, loch, tita, count):
        """To be overriden by some subclasses. s must have 'END' at the end."""
        
        def c_s(node, i, j):
            return self.c_s(pio, s, node, i, j)
        
        n = len(s)
        h = s[loch]
        h_seal = Node('|', h, loch)
        h_l = Node('<', h, loch)
        h_lr = Node('<>', h, loch)
        h_r = Node('>', h, loch)
        h_rl = Node('><', h, loch)
        
        num_stop_left_adj = 0.0
        den_stop_left_adj = c_s(h_l, loch, loch+1)
        for j in range(loch+1, n+1):
            num_stop_left_adj += c_s(h_seal, loch, j)
            den_stop_left_adj += c_s(h_lr, loch, j)
        if self.use_adj:
            tita.add(('stop_left', h, True), num_stop_left_adj)
            count.add(('stop_left', h, True), den_stop_left_adj)
        
        num_stop_left_nonadj = 0.0
        den_stop_left_nonadj = 0.0
        for i in range(0, loch):
            den_stop_left_nonadj += c_s(h_l, i, loch+1)
            for j in range(loch+1, n+1):
                # como i < loch, esta es la fraccion de arboles en las 
                # que h si toma argumento a izquierda...
                num_stop_left_nonadj += c_s(h_seal, i, j)
                den_stop_left_nonadj += c_s(h_lr, i, j)
        if self.use_adj:
            tita.add(('stop_left', h, False), num_stop_left_nonadj)
            count.add(('stop_left', h, False), den_stop_left_nonadj)
        else:
            tita.add(('stop_left', h), num_stop_left_adj+num_stop_left_nonadj)
            count.add(('stop_left', h), den_stop_left_adj+den_stop_left_nonadj)
        
        num_stop_right_adj = 0.0
        den_stop_right_adj = c_s(h_r, loch, loch+1)
        for i in range(loch+1):
            num_stop_right_adj += c_s(h_seal, i, loch+1)
            den_stop_right_adj += c_s(h_rl, i, loch+1)
        if self.use_adj:
            tita.add(('stop_right', h, True), num_stop_right_adj)
            count.add(('stop_right', h, True), den_stop_right_adj)
        
        num_stop_right_nonadj = 0.0
        den_stop_right_nonadj = 0.0
        for j in range(loch+2, n+1):
            den_stop_right_nonadj += c_s(h_r, loch, j)
            for i in range(loch+1):
                num_stop_right_nonadj += c_s(h_seal, i, j)
                den_stop_right_nonadj += c_s(h_rl, i, j)
        if self.use_adj:
            tita.add(('stop_right', h, False), num_stop_right_nonadj)
            count.add(('stop_right', h, False), den_stop_right_nonadj)
        else:
            tita.add(('stop_right', h), num_stop_right_adj+num_stop_right_nonadj)
            count.add(('stop_right', h), den_stop_right_adj+den_stop_right_nonadj)
    
    def num_attach_left(self, pio, s, loch, loca):
        """s must have 'END' at the end."""
        n = len(s)
        (pi, po) = (pio.pi, pio.po)
        h = s[loch]
        h_l = Node('<', h, loch)
        h_lr = Node('<>', h, loch)
        a = s[loca]
        a_seal = Node('|', a, loca)
        end_seal = Node('|', self.end_symbol, n-1)
        """if loch == n-1:
            # es el END, solo cuentan i = 0, j = n-1, k = n
            num_attach_left = pi[0, n-1].val(a_seal) * self.p_attach_left(a, h, loch-loca) / pi[0, n].val(end_seal)
            return num_attach_left"""
        
        num_attach_left = 0.0
        for i in range(loca+1):
            for j in range(loca+1, loch+1):
                num_tmp = po[i, loch+1].val(h_l) * pi[j, loch+1].val(h_l)
                # OPTIMIZACION: solo hasta n-1
                for k in range(loch+1, n):
                    num_tmp += po[i, k].val(h_lr) * pi[j, k].val(h_lr)
                num_attach_left += self.p_nonstop_left(h, j==loch) * pi[i, j].val(a_seal) * num_tmp
        # loch-loca es la distancia entre uno y otro
        num_attach_left *= self.p_attach_left(a, h, loch-loca) / pi[0, n].val(end_seal)
        
        return num_attach_left
    
    def num_attach_right(self, pio, s, loch, loca):
        """s must have 'END' at the end."""
        n = len(s)
        """if loch == n-1:
            # es el END, no me rompas las bolas:
            return 0.0"""
        (pi, po) = (pio.pi, pio.po)
        h = s[loch]
        h_r = Node('>', h, loch)
        h_rl = Node('><', h, loch)
        a = s[loca]
        a_seal = Node('|', a, loca)
        end_seal = Node('|', self.end_symbol, n-1)
    
        num_attach_right = 0.0
        for j in range(loch+1, loca+1):
            # OPTIMIZACION: solo hasta n-1
            for k in range(loca+1, n):
                num_tmp = po[loch, k].val(h_r) * pi[loch, j].val(h_r)
                for i in range(loch+1):
                    num_tmp += po[i, k].val(h_rl) * pi[i, j].val(h_rl)
                num_attach_right += self.p_nonstop_right(h, j==loch+1) * pi[j, k].val(a_seal) * num_tmp
        # loca-loch es la distancia entre uno y otro
        num_attach_right *= self.p_attach_right(a, h, loca-loch) / pi[0, n].val(end_seal)
        
        return num_attach_right
    
    def p_inside_outside(self, s):
        """s must have 'END' at the end."""
        p_inside = self.p_inside(s)
        
        n = len(s)
        p_outside = {}
        # OPTIMIZATION: code the result directly
        #base_list = []
        #node = Node('|', self.end_symbol, n-1)
        #base_list = self.unary_p_outside(1.0, node, 0, n)
        #p_outside[0, n] = PIODict(base_list)
        p_outside[0, n] = PIODict([(1.0, Node('|', self.end_symbol, n-1)), (1.0, Node('<>', self.end_symbol, n-1))])
        
        # OPTIMIZATION: END considered only explicitly
        n = n-1
        
        # OPTIMIZATION: separately compute the first step
        base_list = []
        for i in range(n):
            w = s[i]
            prob = self.p_attach_left(w, self.end_symbol, n-i)
            node = Node('|', w, i)
            base_list += self.unary_p_outside(prob, node, 0, n)
        p_outside[0, n] = PIODict(base_list)
        
        # l goes from n-1 to 1.
        for l in range(n-1, 0, -1):
            for i in range(n-l+1):
                j = i + l
                
                p_outside_dict = PIODict()
                
                for k in range(i):
                    for (p1, n1) in p_inside[k, i].itervalues():
                        for (p2, n2) in p_outside[k, j].itervalues():
                            if n1 == n2 and n1.mark[0] == '>':
                                h = n1.word
                                for m in range(i, j):
                                    a = s[m]
                                    # m-n1.index = distancia entre uno y otro
                                    p3 = self.p_nonstop_right(h, n1.index==i-1) * \
                                        self.p_attach_right(a, h, m-n1.index) * \
                                        p1 * p2
                                    n3 = Node('|', a, m)
                                    p_outside_dict.add(p3, n3)
                            elif n1.mark == '|' and n2.mark[0] == '<' and \
                                                i <= n2.index and n2.index < j:
                                # m-n1.index = distancia entre uno y otro
                                m = n2.index
                                h = n2.word
                                p3 = self.p_nonstop_left(h, m==i) * \
                                    self.p_attach_left(n1.word, h, m-n1.index) * \
                                    p1 * p2
                                n3 = Node(n2.mark, h, m)
                                p_outside_dict.add(p3, n3)
                for k in range(j+1, n+1):
                    for (p1, n1) in p_inside[j, k].itervalues():
                        for (p2, n2) in p_outside[i, k].itervalues():
                            if n1 == n2 and n1.mark[0] == '<':
                                h = n1.word
                                for m in range(i, j):
                                    a = s[m]
                                    # n1.index-m = distancia entre uno y otro
                                    p3 = self.p_nonstop_left(h, n1.index==j) * \
                                        self.p_attach_left(a, h, n1.index-m) * \
                                        p1 * p2
                                    n3 = Node('|', a, m)
                                    p_outside_dict.add(p3, n3)
                            elif n1.mark == '|' and n2.mark[0] == '>' and \
                                                i <= n2.index and n2.index < j:
                                m = n2.index
                                h = n2.word
                                # n1.index-m = distancia entre uno y otro
                                p3 = self.p_nonstop_right(h, m==j-1) * \
                                    self.p_attach_right(n1.word, h, n1.index-m) * \
                                    p1 * p2
                                n3 = Node(n2.mark, h, m)
                                p_outside_dict.add(p3, n3)
                
                if l == 1:
                    end = True
                else:
                    end = False
                p_outside[i, j] = PIODict(sum((self.unary_p_outside(p, node, i, j, end) \
                                for (p, node) in p_outside_dict.itervalues()), []))
        
        return PIO(n+1, p_inside, p_outside, self.end_symbol)
    
    def p_inside(self, s):
        """s must have 'END' at the end."""
        p_inside = {}
        # OPTIMIZACION: esta sutileza hace casi todo:
        #n = len(s)
        n = len(s) - 1
        
        for i in range(n):
            j = i + 1
            w = s[i]
            p = self.p_order('left', w)
            n0 = Node('<', w, i)
            n1 = Node('>', w, i)
            p_inside[i, j] = PIODict(self.unary_p_inside(p, n0, i, j) + \
                                        self.unary_p_inside(1.0 - p, n1, i, j))
        
        for l in range(2, n+1):
            for i in range(n-l+1):
                j = i + l
                # tenemos p_inside[a, b] para todas las cosas adentro de (i, j).
                p_inside_dict = PIODict()
                for k in range(i+1, j):
                    for (p1, n1) in p_inside[i, k].itervalues():
                        for (p2, n2) in p_inside[k, j].itervalues():
                            if n1.mark[0] == '>' and n2.mark == '|':
                                m = n1.index
                                h = n1.word
                                # n2.index-m = distancia entre uno y otro
                                p = self.p_nonstop_right(h, m==k-1) * \
                                    self.p_attach_right(n2.word, h, n2.index-m) * \
                                    p1 * p2
                                p_inside_dict.add(p, n1)
                            if n1.mark == '|' and n2.mark[0] == '<':
                                m = n2.index
                                h = n2.word
                                # m-n1.index = distancia entre uno y otro
                                p = self.p_nonstop_left(h, m==k) * \
                                    self.p_attach_left(n1.word, h, m-n1.index) * \
                                    p1 * p2
                                p_inside_dict.add(p, n2)
                
                # here is where the stops are generated:
                p_inside[i, j] = PIODict(sum((self.unary_p_inside(p, node, i, j) \
                                for (p, node) in p_inside_dict.itervalues()), []))
        
        # OPTIMIZACION: finally, choose the head of the sentence.
        p_sum = 0.0
        for i in range(n):
            w = s[i]
            p1 = p_inside[0, n].val('|'+w+str(i))
            p_sum += p1 * self.p_attach_left(w, self.end_symbol, n-i)
        p_inside_dict = PIODict()
        p_inside_dict.add(p_sum, Node('<', self.end_symbol, n))
        p_inside_dict.add(p_sum, Node('|', self.end_symbol, n))
        p_inside[0, n+1] = p_inside_dict
        
        return p_inside
    
    def parse(self, s):
        t, w = self.dep_parse(s)
        return (treebank.Tree(t), w)
    
    def dep_parse(self, s):
        parse = {}
        # OPTIMIZATION: END considered only explicitly
        # s = s + [self.end_symbol]
        n = len(s)
        
        for i in range(n):
            j = i + 1
            # >w -> w
            # <w -> w
            w = s[i]
            p = self.p_order('left', w)
            t0 = tree.Tree(Node('<', w, i), [w])
            t1 = tree.Tree(Node('>', w, i), [w])
            
            parse[i, j] = ParseDict(self.unary_parses(p, t0, i, j) + \
                                    self.unary_parses(1.0 - p, t1, i, j))
        
        for l in range(2, n+1):
            for i in range(n-l+1):
                j = i + l
                # tenemos parse[a, b] para todas las cosas adentro de (i, j).
                parse_dict = ParseDict()
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
                                p = self.p_nonstop_right(h, m==k-1) * \
                                    self.p_attach_right(n2.word, h, n2.index-m) * \
                                    p1 * p2
                                t = tree.Tree(n1, [t1, t2])
                                parse_dict.add(p, t)
                            if n1.mark == '|' and n2.mark[0] == '<':
                                m = n2.index
                                h = n2.word
                                # m-n1.index = distancia entre uno y otro
                                p = self.p_nonstop_left(h, m==k) * \
                                    self.p_attach_left(n1.word, h, m-n1.index) * \
                                    p1 * p2
                                t = tree.Tree(n2, [t1, t2])
                                parse_dict.add(p, t)
                
                # aca se generan los stops
                parse[i, j] = ParseDict(sum((self.unary_parses(p, t, i, j) \
                                    for (p, t) in parse_dict.itervalues()), []))
        
        # solo falta elegir el head de la oracion:
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
    
    def unary_p_outside(self, p, node, i, j, end=False):
        # XXX: This function is lightly recursive, it may be optimized to an 
        # iterative form.
        h = node.word
        loch = node.index
        radj = loch == j-1
        ladj = loch == i
        if node.mark == '|':
            p2 = self.p_stop_right(h, radj) * p
            n2 = Node('><', h, loch)
            n2.index = node.index
            p3 = self.p_stop_left(h, ladj) * p
            n3 = Node('<>', h, loch)
            n3.index = loch
            res = self.unary_p_outside(p2, n2, i, j, end) + \
                    self.unary_p_outside(p3, n3, i, j, end)
        elif node.mark == '><':
            p2 = self.p_stop_left(h, ladj) * p
            n2 = Node('<', h, loch)
            n2.index = loch
            res = self.unary_p_outside(p2, n2, i, j, end)
        elif node.mark == '<>':
            p2 = self.p_stop_right(h, radj) * p
            n2 = Node('>', h, loch)
            n2.index = loch
            res = self.unary_p_outside(p2, n2, i, j, end)
        elif node.mark == '>' or node.mark == '<':
            if end and node.mark == '>':
                p2 = self.p_order('right', h) * p
                n2 = Node('', h, loch)
                res = [(p2, n2)]
            elif end and node.mark == '<':
                p2 = self.p_order('left', h) * p
                n2 = Node('', h, loch)
                res = [(p2, n2)]
            else:
                res = []
        
        return [(p, node)] + res
    
    def unary_p_inside(self, p, node, i, j):
        # XXX: This function is lightly recursive, it may be optimized to an 
        # iterative form.
        h = node.word
        loch = node.index
        radj = loch == j-1
        ladj = loch == i
        if node.mark == '|':
            res = []
        elif node.mark == '><':
            p2 = self.p_stop_right(h, radj) * p
            n2 = Node('|', h, loch)
            res = [(p2, n2)]
        elif node.mark == '<>':
            p2 = self.p_stop_left(h, ladj) * p
            n2 = Node('|', h, loch)
            res = [(p2, n2)]
        elif node.mark == '>':
            p2 = self.p_stop_right(h, radj) * p
            n2 = Node('<>', h, loch)
            res = self.unary_p_inside(p2, n2, i, j)
        elif node.mark == '<':
            p2 = self.p_stop_left(h, ladj) * p
            n2 = Node('><', h, loch)
            res = self.unary_p_inside(p2, n2, i, j)
        
        return [(p, node)] + res
    
    def unary_parses(self, p, t, i, j):
        # XXX: This function is lightly recursive, it may be optimized to an 
        # iterative form.
        node = t.node
        radj = node.index == j-1
        ladj = node.index == i
        if node.mark == '|':
            res = []
        elif node.mark == '><':
            p2 = self.p_stop_right(node.word, radj) * p
            t2 = tree.Tree(Node('|', node.word, node.index), [t])
            res = [(p2, t2)]
        elif node.mark == '<>':
            p2 = self.p_stop_left(node.word, ladj) * p
            t2 = tree.Tree(Node('|', node.word, node.index), [t])
            res = [(p2, t2)]
        elif node.mark == '>':
            p2 = self.p_stop_right(node.word, radj) * p
            t2 = tree.Tree(Node('<>', node.word, node.index), [t])
            res = self.unary_parses(p2, t2, i, j)
        elif node.mark == '<':
            p2 = self.p_stop_left(node.word, ladj) * p
            t2 = tree.Tree(Node('><', node.word, node.index), [t])
            res = self.unary_parses(p2, t2, i, j)
        
        return [(p, t)] + res
    
    def p_order(self, dir, w):
        #assert w != self.end_symbol
        # OLD! p_order reestimates to 0.5 always!
        if self.tita is None or self.constant_p_order:
            return 0.5
        p = self.tita.val(('order_left', w))
        if dir == 'left':
            return p
        else:
            return 1.0 - p
        #if self.right_first:
        #    if dir == 'right':
        #        return 1.0
        #    else:
        #        return 0.0
        #else:
        #    return 0.5
    
    def p_nonstop_left(self, w, adj):
        return (1.0 - self.p_stop_left(w, adj))
    
    def p_nonstop_right(self, w, adj):
        return (1.0 - self.p_stop_right(w, adj))
    
    def p_stop_left(self, w, adj):
        #assert w != self.end_symbol
        if self.tita is None:
            if adj:
                return 0.25
            else:
                return 0.75
            #return 0.5
        if self.use_adj:
            return self.tita.val(('stop_left', w, adj))
        else:
            return self.tita.val(('stop_left', w))
    
    def p_stop_right(self, w, adj):
        #assert w != self.end_symbol
        if self.tita is None:
            if adj:
                return 0.25
            else:
                return 0.75
            #return 0.5
        if self.use_adj:
            return self.tita.val(('stop_right', w, adj))
        else:
            return self.tita.val(('stop_right', w))
    
    def p_attach_left(self, a, h, dist=None):
        if self.tita is None:
            if h == self.end_symbol:
                # take each word with equal probability. (not normalized)
                return 0.02
            else:
                # not normalized:
                return 1.0 / (1.0 + dist)
        return self.tita.val(('attach_left', a, h))
    
    def p_attach_right(self, a, h, dist=None):
        # assert a != self.end_symbol
        if self.tita is None:
            # not normalized:
            return 1.0 / (1.0 + dist)
        return self.tita.val(('attach_right', a, h))


class Node:
    def __init__(self, mark, word, index):
        self.mark = mark
        self.word = word
        self.index = index
    
    def __eq__(self, other):
        if not isinstance(other, Node): 
            return False
        # XXX: solo haria falta comparar mark e index
        return (self.mark, self.word, self.index) == (other.mark, other.word, other.index)
    
    def __str__(self):
        return str(self.mark) + str(self.word) + str(self.index)
    
    def __repr__(self):
        return self.__str__()


class ParseDict:
    def __init__(self, parses=None):
        self.dict = {}
        if parses is not None:
            self.add_all(parses)
    
    # node puede ser de tipo node o directamente el string
    def val(self, node):
        return self.dict[str(node)]
    
    def add(self, p, t):
        n = t.node
        s = str(n)
        # Aca esta el bias
        # los resultados estan reportados con esta guarda:
        if (s not in self.dict) or (self.dict[s][0] < p):
        #if (s not in self.dict) or (self.dict[s][0] <= p):
            self.dict[s] = (p, t)
    
    def add_all(self, parses):
        for (p, t) in parses:
            self.add(p, t)
    
    def itervalues(self):
        return self.dict.itervalues()


class PIODict(ParseDict):
    default_val = 0.0
    
    def __init__(self, nodes=None, default_val=None):
        self.dict = {}
        if nodes is not None:
            self.add_all(nodes)
        if default_val is not None:
            self.default_val = default_val
    
    # node puede ser de tipo node o directamente el string
    def val(self, n):
        s = str(n)
        if s in self.dict:
            return self.dict[s][0]
        else:
            # print 'No encuentro '+s
            return self.default_val
    
    def add(self, p, n):
        s = str(n)
        if s not in self.dict:
            self.dict[s] = (p, n)
        else:
            self.dict[s] = (self.dict[s][0] + p, n)
    
    def add_all(self, nodes):
        for (p, n) in nodes:
            self.add(p, n)


class PIO:
    def __init__(self, n, pi, po, end_symbol):
        self.n = n
        self.pi = pi
        self.po = po
        self.end_symbol = end_symbol
        # Denominador para c_s:
        self.denom = pi[0, n].val(Node('|', self.end_symbol, n-1))
    
    def c_s(self, node, i, j):
        res = self.pi[i, j].val(node) * self.po[i, j].val(node) / self.denom
        return res
    
    # FIXME: no puede andar porque necesita invocar DMV.p_attach_right(a, h, loca-loch)
    """def num_attach_right(self, s, loch, loca):
        pi, po, n = self.pi, self.po, self.n
        end_seal = Node('|', self.end_symbol, n-1)
        h_r = Node('>', h, loch)
        h_rl = Node('><', h, loch)
        
        num_attach_right = 0.0
        for j in range(loch+1, loca+1):
            for k in range(loca+1, n+1):
                num_tmp = po[loch, k].val(h_r) * pi[loch, j].val(h_r)
                for i in range(loch+1):
                    num_tmp += po[i, k].val(h_rl) * pi[i, j].val(h_rl)
                num_attach_right = pi[j, k].val(a_seal) * num_tmp
        # loch - loca es la distancia entre uno y otro
        num_attach_right *= self.p_attach_right(a, h, loca-loch) / pi[0, n].val(end_seal) 
        
        return num_attach_right"""


class DMVDict(paramdict.ParamDict):
    """Returns KeyError if x is not in the dict.
    """
    def val(self, x):
        return self.d[x]


def test0():
    tb = treebank.Treebank([])
    m = DMV(tb)
    parse = m.parse([])
    #from nltk.draw import draw_trees
    # parse = parse[0,1]
    # draw_trees(*[t for (p, t) in parse.itervalues()])
    return m


def test1():
    tb = treebank.Treebank([])
    m = DMV(tb)
    parse = m.parse(['NNP'])
    #from nltk.draw import draw_trees
    #parse = parse[0,1]
    #draw_trees(*[t for (p, t) in parse.itervalues()])
    return m


def test4():
    """m = DMVTest()
    io = m.p_inside_outside('stocks fell END'.split())
    (i, o) = (io.pi, io.po)
    print "Todos estos numeros deben ser iguales:"
    print 'o[0,1].dict[\'stocks0\']', o[0,1].dict['stocks0']
    print 'o[1,2].dict[\'fell1\']', o[1,2].dict['fell1']
    print 'o[2,3].dict[\'END2\']', o[2,3].dict['END2']
    print 'i[0,3].dict[\'|END2\']', i[0,3].dict['|END2']"""
    m = DMVTest()
    pio = test_p_inside_outside(m, 'stocks fell'.split())
    m = DMV()
    pio = test_p_inside_outside(m, 'stocks stocks'.split())
    return pio


def test_p_inside_outside(m, s):
    s = s + ['END']
    n = len(s)
    pio = m.p_inside_outside(s)
    (pi, po) = (pio.pi, pio.po)
    print "Todos estos numeros deben ser iguales:"
    # after optimizations we do not compute O[n-1,n] nor I[n-1,n], so i < n-1:
    for i in range(n-1):
        print 'O[%i, %i].dict[%s]' % (i, i+1, s[i]), po[i, i+1].dict[s[i]+str(i)]
    print 'I[0, %i].dict[|END]' % n, pi[0, n].dict['|END'+str(n-1)]
    return pio


def test6(ntrees=10, steps=10):
    import wsj10
    tb = wsj10.WSJ10()
    tb.trees = tb.trees[:ntrees]
    m = DMV(tb)
    m.train(steps)
    return m


def test7():
    m = test6(1, 0)
    s = 'stocks fell END'.split()
    pio = m.p_inside_outside(s)
    assert self.c_s(pio, '<>END2', 0, 3) == 1.0
    assert self.c_s(pio, '><END2', 0, 3) == 0.0
    assert self.c_s(pio, '<>END2', 1, 3) == 0.0


def test8():
    m = DMV()
    m.train(10)
    return m
