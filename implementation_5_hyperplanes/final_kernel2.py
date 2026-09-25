import numpy as np, warnings, time, json, pandas as pd; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from data import load_all
from walkkernel import walk_kernel
D=load_all(); cv=StratifiedKFold(5,shuffle=True,random_state=42)
Cs=[0.3,1,3,10,30,100,300]
def K_of(sp,A,B): return rbf_kernel(A,B,gamma=sp[1]) if sp[0]=='rbf' else walk_kernel(A,B,sp[1],sp[2],'product')
arms={'QW product kernel':[('walk',s,n) for n in [3,4,5,6,8] for s in [0.05,0.1,0.2,0.4,0.8,1.2]],
      }
out={}
for a,specs in arms.items():
    rows=[]
    for k in range(2,21):
        Xtr,ytr,Xte,yte=D[k]; best=(-1,None,None)
        for sp in specs:
            K=K_of(sp,Xtr,Xtr)
            for C in Cs:
                acc=np.mean([(SVC(kernel='precomputed',C=C).fit(K[np.ix_(i,i)],ytr[i]).predict(K[np.ix_(j,i)])==ytr[j]).mean() for i,j in cv.split(Xtr,ytr)])
                if acc>best[0]: best=(acc,sp,C)
        _,sp,C=best
        m=SVC(kernel='precomputed',C=C).fit(K_of(sp,Xtr,Xtr),ytr)
        te=(m.predict(K_of(sp,Xte,Xtr))==yte).mean()
        rows.append((k,best[0],te,str(sp),C))
    df=pd.DataFrame(rows,columns=['k','cv','test','spec','C']); out[a]=df
    print(a,'  mean test over 19: %.3f   mean cv %.3f'%(df.test.mean(),df.cv.mean()),flush=True)
    print(df.to_string(index=False),flush=True)
pd.concat(out,names=['arm']).to_csv('kernel_results_run2.csv')
