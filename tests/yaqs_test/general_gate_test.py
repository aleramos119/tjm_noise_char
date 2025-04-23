

#%%

from  mqt.yaqs.core.libraries.gate_library import BaseGate

import numpy as np
#%%


rel=Destroy()

x=X()
y=Y()
z=Z()

jump_list=[rel,z]

obs_list=[x,y,z]


matrices=[]

for lk in jump_list:
    for on in obs_list:
        res=lk.dag()*on*lk  -  0.5*on*lk.dag()*lk  -  0.5*lk.dag()*lk*on
        matrices.append(res.matrix)
        

#%%

matrices[5]

# %%

import qutip as qt



create=qt.create(2)


create.full()
# %%


x=BaseGate.x()
# %%
x.set_sites(0)
# %%
x.sites
# %%

rx=BaseGate.rx(np.pi/2)
# %%
rx.matrix
# %%

rx=BaseGate.rx(np.pi/1)
# %%
rx.matrix
# %%
