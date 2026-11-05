import sys, json, time, numpy as np
import models as m
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
seed=int(sys.argv[1]); gens=int(sys.argv[2])
X,y=m.load(); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=seed)
sx=MinMaxScaler().fit(Xtr); sy=MinMaxScaler().fit(ytr.reshape(-1,1))
t=time.time(); p,k=m.ann_ga(sx.transform(Xtr),sy.transform(ytr.reshape(-1,1)).ravel(),sx.transform(Xte),seed,pop=100,gens=gens)
pred=sy.inverse_transform(np.asarray(p).reshape(-1,1)).ravel()
r=dict(m.metrics(yte,pred),params=int(k),seconds=time.time()-t,gens=gens,pop=100)
json.dump(r,open(f"ga_long_{seed}.json","w")); print(seed,r,flush=True)
