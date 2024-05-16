import numpy as np
import matplotlib.pyplot as plt
import sys

# This function is called in command line with three arguments: num_conditions and num_reactants_1 and num_reactants_2 and creates a fake dataset for testing purposes
# It then saves the dataset to a file calles toy_dataset_{num_conditions}x{num_reactions}.csv
# It also saves an image of the dataset called toy_surface_{num_conditions}x{num_reactions}.png

num_conditions = int(sys.argv[1])
num_block_A = int(sys.argv[2])
num_block_B = num_block_A
try:
    num_block_B = int(sys.argv[3])
except:
    num_block_B = int(sys.argv[3])


hot_spots = [(int(num_conditions/2), int(num_block_A/6), int(num_block_B/6)), (int(num_conditions/4), int(num_block_A/2), int(5*num_block_B/6)), (int(3*num_conditions/4), int(5*num_block_A/6), int(num_block_B/2))]

print(hot_spots)

hot_spots_close_together = ((int(num_block_A/6), int(num_block_B/6), int(num_conditions/2)),(int(num_block_A/2), int(5*num_block_B/6), int(5*num_conditions/8)),(int(5*num_block_A/6), int(num_block_B/2), int(3*num_conditions/4)))

surface = np.zeros((num_conditions, num_block_A, num_block_B))

close_together_surface = np.zeros((num_conditions, num_block_A, num_block_B))

def distance_to_hot_spots(i, j, k):
    distance = num_block_A**2 + num_block_B**2 + num_conditions**2
    for spot in hot_spots:
        distance = min(np.sqrt((i - spot[0])**2 + (j - spot[1])**2 + (k - spot[2])**2), distance)
        if (distance == 0):
            print(distance)
            print(f"{i}, {j}, {k}")
        # print(distance)
    return distance

def create_correlated_surface():
    global surface
    temp_surface = np.zeros((num_conditions, num_block_A, num_block_B))
    for i in range(num_conditions):
        for j in range(num_block_A):
            for k in range(num_block_B):
                temp_surface[i][j][k] = 100 - (distance_to_hot_spots(i, j, k))
    print(np.amin(temp_surface, axis=(0,1,2)))
    temp_surface = temp_surface - np.amin(temp_surface, axis=(0,1,2))
    noise = np.random.normal(0, (np.average(temp_surface, axis=(0,1,2))/4), (num_conditions, num_block_A, num_block_B))
    # print(noise)
    temp_surface = temp_surface + noise
    surface = temp_surface - np.amin(temp_surface, axis=(0,1,2))
    print(np.amax(surface, axis=(0,1,2)))
    surface = surface/np.amax(surface, axis=(0,1,2))

# print(noise)

def write_toy_dataset_to_file():
    with open(f'correlated_toy_{num_conditions}x{num_block_A}x{num_block_B}.csv', 'w+') as f:
        f.write('condition,block_A,block_B,yield\n')
        for i in range(num_conditions):
            for j in range(num_block_A):
                for k in range(num_block_B):
                    f.write(f'{i},{j},{k},{surface[i][j][k]}\n')
    # with open(f'correlated_close_together_toy_{num_conditions}x{num_block_A}x{num_block_B}.csv', 'w') as f:
        # f.write('condition,block_A,block_B,yield\n')
        # for i in range(num_conditions):
            # for j in range(num_block_A):
                # for k in range(num_block_B):
                    # f.write(f'{i},{j},{k},{close_together_surface[i][j][k]}\n')

def write_surface_to_image(surf):
    max_yield_surface = np.zeros((num_block_B, num_block_A))
    surf = surf.T
    print(len(surf))
    print(num_block_A)
    for i in range(num_block_B):
        for j in range(num_block_A):
            max_yield_surface[i][j] += np.amax(surf[i][j])
    max_yield_surface = max_yield_surface.T
    plt.imshow(max_yield_surface)
    plt.colorbar(label='% Yield')
    plt.xlabel('Reactant A')
    plt.ylabel('Reactant B')
    # plt.zlabel('Conditions')
    plt.title(f'Toy Surface with {num_conditions} Conditions and {num_block_A}x{num_block_B} Reactions')
    plt.savefig(f'correlated_toy_{num_conditions}x{num_block_A}x{num_block_B}_1.png')
    return

create_correlated_surface()
write_toy_dataset_to_file()
write_surface_to_image(surface)

def create_image_from_csv():
    temp_surface = np.zeros((num_conditions, num_block_A, num_block_B))
    with open(f'correlated_toy_{num_conditions}x{num_block_A}x{num_block_B}.csv', 'r') as f:
        lines = f.readlines()[1:]
        
        for line in lines:
            line = line.split(',')
            temp_surface[int(line[0])][int(line[1])][int(line[2])] = float(line[3])
    write_surface_to_image(temp_surface)

# create_image_from_csv()

# create_correlated_surface()