"""
dqn_agent.py  (NumPy-only implementation)
------------------------------------------
Deep Q-Network (Double DQN) agent in pure NumPy.
No deep-learning framework required.

Architecture:  in → FC(128) → ReLU → FC(128) → ReLU → FC(64) → ReLU → FC(action_size)
"""

import random, pickle
import numpy as np
from collections import deque

# ---- Activations ----
def relu(x):      return np.maximum(0, x)
def relu_g(x):    return (x > 0).astype(np.float32)
def huber(y,t,d=1.0):
    e=y-t; a=np.abs(e)
    return np.where(a<=d, .5*e**2, d*(a-.5*d))
def huber_g(y,t,d=1.0):
    e=y-t; a=np.abs(e)
    return np.where(a<=d, e, d*np.sign(e))

class Dense:
    def __init__(self, ni, no, lr=1e-3):
        s = np.sqrt(2/ni)
        self.W  = (np.random.randn(ni,no)*s).astype(np.float32)
        self.b  = np.zeros(no, np.float32)
        self.lr = lr
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b)
        self.t=0; self._x=None

    def fwd(self, x):
        self._x=x; return x@self.W+self.b

    def bwd(self, g, clip=1.0):
        x=self._x; B=len(x)
        dW=x.T@g/B; db=g.mean(0); dx=g@self.W.T
        n=np.sqrt((dW**2).sum()+(db**2).sum())+1e-8
        if n>clip: dW*=clip/n; db*=clip/n
        b1,b2,eps=0.9,0.999,1e-8; self.t+=1
        self.mW=b1*self.mW+(1-b1)*dW; self.vW=b2*self.vW+(1-b2)*dW**2
        self.mb=b1*self.mb+(1-b1)*db; self.vb=b2*self.vb+(1-b2)*db**2
        mW=self.mW/(1-b1**self.t); vW=self.vW/(1-b2**self.t)
        mb=self.mb/(1-b1**self.t); vb=self.vb/(1-b2**self.t)
        self.W-=self.lr*mW/(np.sqrt(vW)+eps)
        self.b-=self.lr*mb/(np.sqrt(vb)+eps)
        return dx

    def predict(self, x): return x@self.W+self.b
    def get_w(self): return (self.W.copy(), self.b.copy())
    def set_w(self, W, b): self.W=W.copy(); self.b=b.copy()
    def soft_upd(self, s, tau=0.005):
        self.W=tau*s.W+(1-tau)*self.W; self.b=tau*s.b+(1-tau)*self.b


class QNetwork:
    def __init__(self, si, ai, hidden=128, lr=1e-3):
        self.L=[Dense(si,hidden,lr), Dense(hidden,hidden,lr),
                Dense(hidden,64,lr), Dense(64,ai,lr)]
        self._pre=[]

    def forward(self, x):
        self._pre=[]; h=x.astype(np.float32)
        for i,l in enumerate(self.L):
            h=l.fwd(h)
            if i<len(self.L)-1: self._pre.append(h.copy()); h=relu(h)
        return h

    def backward(self, g):
        g=g.astype(np.float32)
        for i in range(len(self.L)-1,-1,-1):
            g=self.L[i].bwd(g)
            if i>0: g=g*relu_g(self._pre[i-1])

    def predict(self, x):
        h=x.astype(np.float32)
        for i,l in enumerate(self.L):
            h=l.predict(h)
            if i<len(self.L)-1: h=relu(h)
        return h

    def copy_from(self, src):
        for t,s in zip(self.L,src.L): t.set_w(*s.get_w())
    def soft_upd_from(self, src, tau=0.005):
        for t,s in zip(self.L,src.L): t.soft_upd(s,tau)


class ReplayBuffer:
    def __init__(self, cap=50_000): self.buf=deque(maxlen=cap)
    def push(self,s,a,r,ns,d):
        self.buf.append((np.array(s,np.float32),int(a),float(r),
                         np.array(ns,np.float32),bool(d)))
    def sample(self, bs):
        b=random.sample(self.buf,bs); s,a,r,ns,d=zip(*b)
        return (np.stack(s),np.array(a),np.array(r,np.float32),
                np.stack(ns),np.array(d,np.float32))
    def __len__(self): return len(self.buf)


class DQNAgent:
    """Double DQN trading agent (pure NumPy, no framework required)."""

    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, buffer_size=50_000, tau=0.005,
                 update_every=4, hidden=128):
        self.state_size=state_size; self.action_size=action_size
        self.gamma=gamma; self.epsilon=epsilon
        self.epsilon_min=epsilon_min; self.epsilon_decay=epsilon_decay
        self.batch_size=batch_size; self.tau=tau; self.update_every=update_every

        self.q  = QNetwork(state_size, action_size, hidden, lr)
        self.qt = QNetwork(state_size, action_size, hidden, lr)
        self.qt.copy_from(self.q)

        self.mem=ReplayBuffer(buffer_size)
        self.step_count=0; self.losses=[]

    def act(self, state, training=True):
        if training and random.random()<self.epsilon:
            return random.randrange(self.action_size)
        return int(self.q.predict(state[np.newaxis,:])[0].argmax())

    def remember(self, s, a, r, ns, d):
        self.mem.push(s,a,r,ns,d); self.step_count+=1
        if len(self.mem)>=self.batch_size and self.step_count%self.update_every==0:
            self.losses.append(self._learn())

    def _learn(self):
        S,A,R,NS,D=self.mem.sample(self.batch_size); B=len(S)
        q_all=self.q.forward(S)
        best_a=self.q.predict(NS).argmax(1)
        qt_ns =self.qt.predict(NS)
        tgt=R+self.gamma*qt_ns[np.arange(B),best_a]*(1-D)
        q_tgt=q_all.copy(); q_tgt[np.arange(B),A]=tgt
        loss=huber(q_all,q_tgt).mean()
        self.q.backward(huber_g(q_all,q_tgt)/B)
        self.qt.soft_upd_from(self.q,self.tau)
        return float(loss)

    def decay_epsilon(self):
        self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)

    def save(self, path="dqn_trading_agent.pkl"):
        with open(path,"wb") as f:
            pickle.dump({"q"  :[(l.W,l.b) for l in self.q.L],
                         "qt" :[(l.W,l.b) for l in self.qt.L],
                         "eps":self.epsilon,"steps":self.step_count},f)
        print(f"[Agent] Saved → {path}")

    def load(self, path="dqn_trading_agent.pkl"):
        with open(path,"rb") as f: d=pickle.load(f)
        for l,(W,b) in zip(self.q.L,d["q"]): l.set_w(W,b)
        for l,(W,b) in zip(self.qt.L,d["qt"]): l.set_w(W,b)
        self.epsilon=d["eps"]; self.step_count=d["steps"]
        print(f"[Agent] Loaded ← {path}")
