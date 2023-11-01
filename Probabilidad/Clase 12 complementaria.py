import numpy as np


N=2000000
todos_tuvieron_premio = 0
dos_tuvieron_premio = 0
uno_tuvo_premio = 0
ninguno_tuvo_premio = 0
for i in range(N):
    boletas = np.zeros(50)
    boletas[:3] = 1
    
    np.random.shuffle(boletas)
    organizadores = boletas[:4]
    if sum(organizadores) == 3:
        todos_tuvieron_premio += 1
    if sum(organizadores) == 2:
        dos_tuvieron_premio +=1 
    if sum(organizadores) == 1:
        uno_tuvo_premio += 1
    if sum(organizadores) == 0:
        ninguno_tuvo_premio += 1


print(f'Probabilidad de que:')
print(f'Todos tuvieron premio = {todos_tuvieron_premio/N}')
print(f'Solo dos tuvieron premio = {dos_tuvieron_premio/N}')
print(f'Solo uno tuvo premio = {uno_tuvo_premio/N}')
print(f'Ninguno tuvo premio = {ninguno_tuvo_premio/N}')
        
        