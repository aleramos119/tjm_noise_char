

#%%

from  mqt.yaqs.core.libraries.gate_library import *


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
