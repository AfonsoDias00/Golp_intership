import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Step 1: Read data from text file
data = np.loadtxt('total_current')
data2 = np.loadtxt('laser')

# Extract time and current data from columns
time = data[:, 1]

current_x = data[:, 2]
current_y = data[:, 3]
current_z = data[:, 4]
A1 = data[:, 2]
modified_A = list(map(lambda x: 10*x + 20, A1))

# Step 2: Plot current over time
plt.figure(figsize=(10, 6))
plt.plot(time*0.6582, current_x, label='Electric Current') #0.6582 factor from /eV to fs
plt.xlabel('Time (fs)')
plt.ylabel('Electric Current (arb.)')
plt.title('Electric Current over Time')
plt.grid(True)
plt.legend()
plt.savefig('Current.png')

    
# Step 3: Perform Fourier Transform
# Compute the FFT
fft_resultx = np.fft.fft(current_x)
fft_resulty = np.fft.fft(current_y)
fft_resultz = np.fft.fft(current_z)

result = np.abs(fft_resultx) ** 2 +  np.abs(fft_resulty) ** 2 +  np.abs(fft_resultz) ** 2

# Step 4: Plot spectral frequencies
# Compute the frequencies
n = len(time)
freq = np.fft.fftfreq(n, d=(time[1]-time[0]))

# Find the indices of positive frequencies
positive_indices = np.where(freq > 0)

# Select positive frequencies and corresponding Fourier transform values
positive_omega = 2*np.pi*freq[positive_indices]

positive_fft_result = result[positive_indices]
# Plot the spectrum
'''
plt.figure(figsize=(10, 6))
plt.plot(positive_omega, np.abs(positive_fft_result))
plt.vlines(np.arange(0, 41, 1.55), ymin=0, ymax=np.max(np.abs(positive_fft_result)), color='darkgray', linestyle='--')
plt.xlim(0, 35)
plt.ylim(10**-2, 10**6)
plt.xlabel('Frequency /eV')
plt.ylabel('Spectral Intensity (arb.)')
plt.yscale('log')
plt.title('Harmonic order')
plt.suptitle('Harmonic Spectrum', fontsize=16, y=0.98)
#plt.grid(True)
plt.savefig('Spectral_current.png')
'''

# Plot the spectrum
plt.figure(figsize=(10, 6))
plt.plot(positive_omega, np.abs(positive_fft_result))

# Define the positions of the vertical lines
vline_positions = np.arange(1.55, 35, 1.55)

# Plot the vertical lines
plt.vlines(vline_positions, ymin=0, ymax=np.max(np.abs(positive_fft_result)), color='darkgray', linestyle='--')

# Add the numbers above each vertical line, but only up to 23
for i, position in enumerate(vline_positions):
    
    plt.text(position, np.max(np.abs(positive_fft_result)) * 1.1, str(i + 1), ha='center', va='bottom', color='black')

plt.xlim(0, 35)
plt.ylim(10**-2, 10**6)
plt.xlabel('Frequency /eV')
plt.ylabel('Spectral Intensity (arb.)')
plt.yscale('log')
plt.title('Harmonic order', fontsize=12, y=1.05)  # Move the title slightly up


plt.savefig('Spectral_current.png')


#---------------------- Time-frequency analysis------------------------------

#time *= 0.6582
def gabor_transform(tau, omega):
    # Define Gaussian window
    w = 1.3  # Based on the signal characteristics
    window = np.exp(-0.5 * ((time - tau) / w) ** 2)
    e1 = np.cos(omega*time)
    e2 = np.sin(omega*time)

    d_current_x = np.gradient(current_x, time)
    integral_x1 = trapz(window*e1*d_current_x, time)
    integral_x2 = trapz(window*e2*d_current_x, time)
    integral_x = integral_x1 + 1j*integral_x2
    
    d_current_y = np.gradient(current_y, time)
    integral_y1 = trapz(window*e1*d_current_y, time)
    integral_y2 = trapz(window*e2*d_current_y, time)
    integral_y = integral_y1 + 1j*integral_x2
    
    d_current_z = np.gradient(current_z, time)
    integral_z1 = trapz(window*e1*d_current_z, time)
    integral_z2 = trapz(window*e2*d_current_z, time)
    integral_z = integral_z1 + 1j*integral_z2
    
    integral = np.sum(np.abs(integral_x)**2 + np.abs(integral_y)**2 + np.abs(integral_z)**2)

    return integral

# Plot spectrogram
#tau = time[positive_indices]
#omega = positive_omega

tau = np.linspace(60, 80, 100)
omega = np.linspace(10, 3000, 100)

# Create a grid of tau and omega values using meshgrid
Tau, Omega = np.meshgrid(tau, omega)

# Initialize array to store results
Z = np.zeros_like(Tau)

# Iterate over each pair of tau and omega values
for i in range(len(omega)):
    for j in range(len(tau)):
        Z[i, j] = gabor_transform(Tau[i, j], Omega[i, j])


print(np.log(Z.min()))
print(np.log(Z.max()))
#cmap = plt.get_cmap('rainbow')
# Define a custom colormap that combines white for values below -15 and a rainbow colormap for values from -15 to 0
cmap = LinearSegmentedColormap.from_list('custom_cmap', [ (0.00, 'blue'), (0.25, 'cyan'), (0.5, 'green'), (0.75, 'yellow'), (1, 'red')], N=256)



# Add a small epsilon to avoid log(0) issues
epsilon = 1e-10
Z_log = -np.log(Z + epsilon) / np.log(Z.min() + epsilon)

# Normalize the data to the range 0 to 1
Z_normalized = (Z_log - Z_log.min()) / (Z_log.max() - Z_log.min())

# Normalize the data to the range -1 to 0
norm = Normalize(vmin=0, vmax=1)
#levels = np.linspace(np.log(Z.min()), np.log(Z.max()), 100)  # Define more levels for higher resolution
levels = np.linspace(0, 1, 200)  # Define more levels for higher resolution
# Create a 2D colored plot
plt.figure(figsize=(8, 6))
plt.contourf(Tau, Omega, Z_normalized, levels=levels, cmap=cmap, norm=norm)  # Use contourf for filled contour plot
#plt.plot(time, modified_A, color='red', label='A(t)')
plt.colorbar(label='Gabor Transform Value (Normalized)')  # Add colorbar to show the scale
plt.xlabel('Time (fs)')
plt.ylabel('Energy (eV)')
plt.title('2D Colored Plot of Gabor Transform')

# Add horizontal lines and labels
harmonics = np.arange(1.55, 40, 3.1)
#odd_harmonics = harmonics[harmonics % 2 == 1]
#for harmonic in harmonics:
#    plt.hlines(harmonic, xmin=0, xmax=80, color='darkgray', linestyle='--')
 #   plt.text(81, harmonic, f'{int(2*((harmonic-1.55)/3.1)+1)}', verticalalignment='center', color='black')
#plt.text(81, 45, 'Harmonic', verticalalignment='center', rotation=90, color='black')
#plt.hlines(np.arange(1.55, 40, 3.1), xmin=0, xmax=80, color='darkgray', linestyle='--')
plt.savefig('Gabor_Spectrogram.png')


