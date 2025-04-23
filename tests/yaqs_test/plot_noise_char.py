
#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
N_list=[500,1000,2000,4000,8000]
samp_list=[50]

file_list=["loss.txt","gamma_rel.txt","gamma_deph.txt"]



#%%
func_list=["mean","std"]

for N in N_list:


    folder=f"noise_char/N_{N}/samples_50/"

    for file in file_list:

        full_file=folder+file
        data=np.genfromtxt(full_file)

        for func in func_list:

            if func=="mean":
                data_1=np.array([[data[i,0],data[i,1], np.mean(data[i,2:])] for i in range(len(data))])

            if func=="std":
                data_1=np.array([[data[i,0],data[i,1], np.std(data[i,2:])] for i in range(len(data))])
        
            plt.figure(figsize=(8, 6))
            contour = plt.tricontourf(data_1[:,0], data_1[:,1], data_1[:,2], levels=100, cmap='viridis')
            plt.colorbar(contour, label=f"{func}")
            plt.xlabel("X-axis (Column 1)")
            plt.ylabel("Y-axis (Column 2)")
            plt.title(f"{file.split('.')[0]}")
            output_file = f"{folder}{file.split('.')[0]}_{func}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()


#%%


for file in file_list:
    std_list_max=[]
    std_list_min=[]
    std_list_avg=[]

    for N in N_list:
        folder=f"noise_char/N_{N}/samples_50/"

        full_file=folder+file
        data=np.genfromtxt(full_file)

        std_list_max.append(np.max(np.std(data[:,2:],axis=1)))
        std_list_min.append(np.min(np.std(data[:,2:],axis=1)))
        std_list_avg.append(np.mean(np.std(data[:,2:],axis=1)))

    plt.figure(figsize=(8, 6))
    plt.plot(N_list,std_list_max,'o-',label='max')
    plt.plot(N_list,std_list_min,'o-',label='min')
    plt.plot(N_list,std_list_avg,'o-',label='avg')
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("std")
    plt.title(f"{file.split('.')[0]}")
    output_file = f"noise_char/{file.split('.')[0]}_std.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    

# %%
full_file=f"noise_char/N_{4000}/samples_50/gamma_deph.txt"

data=np.genfromtxt(full_file)


plt.figure(figsize=(8, 6))
plt.hist(data[0, 2:], bins=10, color='blue', alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of Values")
plt.grid(True)
plt.tight_layout()
# %%



# %%
