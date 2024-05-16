import numpy as np
import matplotlib.pyplot as plt
import sys

# This function is called in command line with two arguments: num_conditions and num_reactions and creates a fake dataset for testing purposes
# It then saves the dataset to a file calles toy_dataset_{num_conditions}x{num_reactions}.csv
# It also saves an image of the dataset called toy_surface_{num_conditions}x{num_reactions}.png

num_conditions = int(sys.argv[1])
num_reactions = int(sys.argv[2])

surface = np.random.random((num_conditions,num_reactions))

def write_toy_dataset_to_file():
    with open(f'toy_dataset_{num_conditions}x{num_reactions}.csv', 'w') as f:
        f.write('condition,reaction,yield\n')
        for i in range(num_conditions):
            for j in range(num_reactions):
                f.write(f'{i},{j},{surface[i][j]}\n')

def write_surface_to_image():
    plt.imshow(surface)
    plt.colorbar(label='% Yield')
    plt.xlabel('Reactions')
    plt.ylabel('Conditions')
    plt.title(f'Toy Surface with {num_conditions} Conditions and {num_reactions} Reactions')
    plt.savefig(f'toy_surface_{num_conditions}x{num_reactions}.png')
    return

write_toy_dataset_to_file()
write_surface_to_image()