#!/usr/bin/env python3
"""Checks the relation between Theorem F(2) and the Schur complement T_slow of Andreyev et al. (2026, App. D).

Claim: I - T_slow = S o L, with L(Sigma) = eta/(1-beta) * [left side of Theorem F(2)] and
S^{-1}(X) = X + beta*eta/((1-beta)^2 (1+beta)) * (H X + X H), independent of the curvature noise.

(1) Exact rational arithmetic, per eigen-pair (j, k): det(I - M22) in closed form, the deterministic part
    (1 - T_slow) = S_jk L_jk, and the noise multiplier eta^2 + M12 (I - M22)^{-1} (-eta, -eta, 1) = S_jk eta^2/(1-beta)^2.
    After clearing denominators these are polynomial identities of degree <= 10 in eta, <= 16 in beta and <= 5 in each
    lambda; agreement on a 13 x 22 x 13 x 13 grid proves them.
(2) Floating point, P = 2, six non-commuting PSD per-sample Hessians: the full operator identity.
(3) Leading term of their remainder: R(Sigma) = 2 beta eta^2/(1-beta)^3 H Sigma H + O(eta^3).
"""
from fractions import Fraction as F
import itertools, random

def det3(M):
    return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
            +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))

def adj3(M):
    C=[[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            m=[[M[r][s] for s in range(3) if s!=j] for r in range(3) if r!=i]
            C[i][j]=((-1)**(i+j))*(m[0][0]*m[1][1]-m[0][1]*m[1][0])
    return [[C[j][i] for j in range(3)] for i in range(3)]

def pair_blocks(eta,b,hj,hk):
    Aj=[[1-eta*hj,-eta*b],[hj,b]]; Ak=[[1-eta*hk,-eta*b],[hk,b]]
    idx=[(0,0),(0,1),(1,0),(1,1)]
    T=[[Aj[a][c]*Ak[a2][d] for (c,d) in idx] for (a,a2) in idx]
    IM=[[(1 if i==j else 0)-T[i+1][j+1] for j in range(3)] for i in range(3)]
    return T,IM

def check_pair(eta,b,hj,hk):
    T,IM=pair_blocks(eta,b,hj,hk)
    M11=T[0][0]; M12=T[0][1:]; M21=[T[i][0] for i in range(1,4)]
    D=det3(IM); Ad=adj3(IM); q=1-b
    s_den=q*q*(1+b)+b*eta*(hj+hk)
    L=eta/q*((hj+hk)-eta*hj*hk/(1+b)+eta*b*(hj-hk)**2/(q*(1-b*b)))
    det_ok=(D==q*s_den)
    lhs=D*(1-M11)-sum(M12[i]*sum(Ad[i][j]*M21[j] for j in range(3)) for i in range(3))
    det_part_ok=(lhs==D*q*q*(1+b)*L/s_den)
    n=[-eta,-eta,1]
    K=eta*eta+sum(M12[i]*sum(Ad[i][j]*n[j] for j in range(3)) for i in range(3))/D
    noise_ok=(K==(1+b)/s_den*eta*eta)
    return det_ok and det_part_ok and noise_ok

def exact_grid():
    E=[F(k,7) for k in range(1,14)]; B=[F(k,23) for k in range(0,22)]; H=[F(k,5) for k in range(0,13)]
    pts=list(itertools.product(E,B,H,H))
    bad=sum(1 for p in pts if not check_pair(*p))
    return len(pts),bad

def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def add(A,B,s=1.0): return [[A[i][j]+s*B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
def inv(A):
    n=len(A); M=[A[i][:]+[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        p=max(range(i,n),key=lambda r:abs(M[r][i])); M[i],M[p]=M[p],M[i]
        piv=M[i][i]; M[i]=[x/piv for x in M[i]]
        for r in range(n):
            if r!=i:
                f=M[r][i]; M[r]=[a-f*b for a,b in zip(M[r],M[i])]
    return [row[n:] for row in M]

def operator_check(P=2,eta=0.05,beta=0.8,seed=7):
    random.seed(seed)
    Hs=[]
    for _ in range(6):
        a=[[random.gauss(0,1) for _ in range(P)] for _ in range(P)]
        Hs.append(mm(a,[list(r) for r in zip(*a)]))
    w=len(Hs); H=[[sum(h[i][j] for h in Hs)/w for j in range(P)] for i in range(P)]
    n=2*P; T=[[0.0]*(n*n) for _ in range(n*n)]
    for Ht in Hs:
        A=[[0.0]*n for _ in range(n)]
        for i in range(P):
            for j in range(P):
                A[i][j]=(1.0 if i==j else 0.0)-eta*Ht[i][j]; A[P+i][j]=Ht[i][j]
            A[i][P+i]=-eta*beta; A[P+i][P+i]=beta
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        T[a*n+b][c*n+d]+=A[a][c]*A[b][d]/w
    u=[a*n+b for a in range(P) for b in range(P)]; y=[k for k in range(n*n) if k not in u]
    sub=lambda R,C: [[T[i][j] for j in C] for i in R]
    IM22=add([[1.0 if i==j else 0.0 for j in range(len(y))] for i in range(len(y))],sub(y,y),-1)
    Ts=add(sub(u,u),mm(mm(sub(u,y),inv(IM22)),sub(y,u)))
    ImTs=add([[1.0 if i==j else 0.0 for j in range(P*P)] for i in range(P*P)],Ts,-1)
    c=beta*eta/((1-beta)**2*(1+beta)); err=0.0
    for k in range(P*P):
        Ek=[[0.0]*P for _ in range(P)]; Ek[k//P][k%P]=1.0
        X=[[ImTs[i*P+j][k] for j in range(P)] for i in range(P)]
        left=add(X,add(mm(H,X),mm(X,H)),c)
        HS=mm(H,Ek); SH=mm(Ek,H); D=add(HS,SH,-1)
        comm=add(mm(H,D),mm(D,H),-1)
        V=[[0.0]*P for _ in range(P)]
        for Ht in Hs:
            Dh=add(Ht,H,-1); V=add(V,mm(mm(Dh,Ek),Dh),1.0/w)
        Lk=add(add(add(add(HS,SH),mm(HS,H),-eta/(1+beta)),comm,eta*beta/((1-beta)*(1-beta**2))),V,-eta/(1-beta))
        Lk=[[eta/(1-beta)*x for x in r] for r in Lk]
        err=max(err,max(abs(left[i][j]-Lk[i][j]) for i in range(P) for j in range(P)))
    return err

def remainder_ratio(eta,b,hj,hk):
    T,IM=pair_blocks(eta,b,hj,hk)
    D=det3(IM); Ad=adj3(IM)
    Ts=T[0][0]+sum(T[0][1+i]*sum(Ad[i][j]*T[1+j][0] for j in range(3)) for i in range(3))/D
    base=1-eta/(1-b)*(hj+hk)+eta**2/(1-b)**2*hj*hk
    return (Ts-base)/(2*b*eta**2*hj*hk/(1-b)**3)

def lemma_remainder_exact():
    """Lemma (remainder of the Schur-complement expansion), deterministic part, exact rational arithmetic:
    R = 2 b eta^2/(1-b)^3 l_j l_k + c K L2 - c^2 K^2 S L entry by entry, and R = 0 when l_k = 0."""
    bad=0; n=0
    for eta in (F(1,10),F(1,37),F(2,3),F(1,1000)):
        for b in (F(0),F(1,2),F(9,10)):
            for hj,hk in ((F(1),F(1)),(F(1),F(1,100)),(F(3),F(0)),(F(2),F(5,7)),(F(1),F(1,1000))):
                T,IM=pair_blocks(eta,b,hj,hk); D=det3(IM); Ad=adj3(IM)
                Ts=T[0][0]+sum(T[0][1+i]*sum(Ad[i][j]*T[1+j][0] for j in range(3)) for i in range(3))/D
                ee=eta/(1-b); R=Ts-(1-ee*(hj+hk)+ee**2*hj*hk)
                K=hj+hk; cc=b*eta/((1-b)**2*(1+b))
                L2=-eta**2/(1-b**2)*hj*hk+eta**2*b/((1-b)**2*(1-b**2))*(hj-hk)**2
                L=ee*K+L2; S=1/(1+cc*K)
                n+=1
                if R!=2*b*eta**2/(1-b)**3*hj*hk+cc*K*L2-cc**2*K**2*S*L: bad+=1
                if hk==0 and R!=0: bad+=1
    return n,bad

if __name__=='__main__':
    n,bad=exact_grid(); print(f'(1) exact grid: {n} points, {bad} failures')
    print(f'(2) operator identity, P=2, non-commuting noise: max residual {operator_check():.1e}')
    # (3) off the diagonal the HSigmaH reading and the double-commutator reading differ:
    # (H S H)_jk = l_j l_k S_jk, [H,[H,S]]_jk = (l_j - l_k)^2 S_jk; only the first tends to 1.
    for b in (F(1,2),F(9,10)):
        for r in (F(1,3),F(1,10**2),F(1,10**3)):
            hss=[round(float(remainder_ratio(F(1,10**e),b,F(1),r)),6) for e in (4,5,6,7)]
            com=[]
            for e in (4,5,6,7):
                eta=F(1,10**e); T,IM=pair_blocks(eta,b,F(1),r); D=det3(IM); Ad=adj3(IM)
                Ts=T[0][0]+sum(T[0][1+i]*sum(Ad[i][j]*T[1+j][0] for j in range(3)) for i in range(3))/D
                Rv=Ts-(1-eta/(1-b)*(1+r)+eta**2/(1-b)**2*r)
                com.append(float(Rv/(b*eta**2*(1-r)**2/((1-b)**3*(1+b)))))
            print(f'(3) beta={float(b)} l_k/l_j={float(r):.0e}: R/HSigmaH-term at eta=1e-4..1e-7 {hss};'
                  f' R/commutator-term {[round(x,6) for x in com]}')
        T,IM=pair_blocks(F(1,10**6),b,F(1),F(0)); D=det3(IM); Ad=adj3(IM)
        Ts=T[0][0]+sum(T[0][1+i]*sum(Ad[i][j]*T[1+j][0] for j in range(3)) for i in range(3))/D
        print(f'(3) beta={float(b)} l_k=0: R = {Ts-(1-F(1,10**6)/(1-b))}')
    n,bad=lemma_remainder_exact(); print(f'(4) exact remainder formula of the lemma: {n} points, {bad} failures')
