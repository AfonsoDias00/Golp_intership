import numpy as np

# Step 1: Read data from text file
data = np.loadtxt('total_current')

time = data[:, 1]
#time *= 0.6582
current_x = data[:, 2]
current_y = data[:, 3]
current_z = data[:, 4]

#tau = np.linspace(0, 90, 1000)
#omega = np.linspace(0, 30, 1000)

def WriteVector(vector1, vector2, vector3, vector4, filename):
    with open(filename, 'w') as file:
        max_length = max(len(vector1), len(vector3))  # Determine the maximum length among all vectors
        for i in range(max_length):
            if i < len(vector1):
                v1 = vector1[i]
            else:
                v1 = ''  # If vector1 has fewer elements, use an empty string

            if i < len(vector2):
                v2 = vector2[i]
            else:
                v2 = ''  # If vector2 has fewer elements, use an empty string

            if i < len(vector3):
                v3 = vector3[i]
            else:
                v3 = ''  # If vector3 has fewer elements, use an empty string

            if i < len(vector4):
                v4 = vector4[i]
            else:
                v4 = ''  # If vector4 has fewer elements, use an empty string

            file.write(f"{v1} {v2} {v3} {v4}\n")


WriteVector(time, current_x, current_y, current_z, "eixos.txt")
