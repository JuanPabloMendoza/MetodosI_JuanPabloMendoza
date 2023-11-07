import numpy as np

Dict = { 0: 'a', 1: 'o', 2:'m', 3:'r', 4:' ' }
T = np.array([[0.00,0.05,0.30,0.30,0.10],\
              [0.10,0.00,0.30,0.30,0.10],\
              [0.30,0.30,0.00,0.00,0.10],\
              [0.20,0.40,0.00,0.00,0.10],\
              [0.20,0.20,0.00,0.40,0.00],\
             ])

def GetString(Initial,T,N=int(30)):
    
    tex = ""
    States = np.array(Initial)
    
    for i in range(N):
        Initial = np.dot(T,Initial)
        States = np.vstack((States,Initial))
        index = np.where(np.amax(Initial)==Initial)[0]
        
        word = Dict[index[0]]
        tex = tex + word
        
    return tex,States

#escribir amor
initial = np.array([0.,0.,1.,1.,0.])
tex,States = GetString(initial, T)
print(f'\n{tex}')