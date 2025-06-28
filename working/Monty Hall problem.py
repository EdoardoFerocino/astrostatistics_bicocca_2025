import numpy as np

def MontyHall(p,N):
    Wins = np.array([0,0,0])
    Iter = 10000
    for _ in range(Iter):
        # Define the doors
        Doors = np.array(["Car"]+["Goat"]*(N-1))
        # np.random.shuffle(Doors)
        
        # The player chooses one door but does not open it
        FirstChoiceDoorID = np.random.randint(0, N)
        
        # The presenter randomly chooses between one of the Goat doors and opens it
        goat_indices = np.where(Doors == "Goat")[0]
        valid_goat_indices = goat_indices[goat_indices != FirstChoiceDoorID]
        GoatDoorID = np.random.choice(valid_goat_indices, size = p, replace = False)
        # replace = False. Whether the sample is with or without replacement. Default is True, meaning that a value of a can be selected multiple times.
        
        # The player now chooses his strategy
        SecondChoiceDoor =  np.full(3, np.nan)
        # Conservative
        SecondChoiceDoor[0] = FirstChoiceDoorID
        
        # Switcher
        all_indices = np.arange(N)
        exclude_indices = np.append(GoatDoorID,FirstChoiceDoorID)
        SecondChoiceDoor[1] = np.random.choice(np.setdiff1d(all_indices, exclude_indices))
        
        # New comer
        valid_newcomerchoice_indices = np.delete(all_indices,GoatDoorID)
        SecondChoiceDoor[2] = np.random.choice(valid_newcomerchoice_indices)
        
        
        Winner = np.array(SecondChoiceDoor == np.where(Doors == "Car")[0])
        Wins[Winner] +=1
    return Wins / Iter

p = 1 #number of doors chosen by the host
N = p+2 # number of doors(p+1 car+1 goat)

print(MontyHall(p,N))


