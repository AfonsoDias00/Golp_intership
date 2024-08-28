import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Step 1: Read data from text file
data = np.loadtxt('total_current')

# Extract time and current data from columns
time = data[:, 1]
#time *=0.6582

current_x = data[:, 2]
current_y = data[:, 3]
current_z = data[:, 4]
d_current_x = np.gradient(current_x, time)
d_current_y = np.gradient(current_y, time)
d_current_z = np.gradient(current_z, time)

# Step 3: Perform Fourier Transform
# Compute the FFT
fft_resultx = np.fft.fft(d_current_x)
fft_resulty = np.fft.fft(d_current_y)
fft_resultz = np.fft.fft(d_current_z)

resultx_shift = np.fft.fftshift(fft_resultx)
resulty_shift = np.fft.fftshift(fft_resulty)
resultz_shift = np.fft.fftshift(fft_resultz)


#omega_0= 1.55103/(2*np.pi)
omega_0 = 1.55103

std_dev = 0.32 * 1.55103 / (2*np.pi)

# Frequency array corresponding to the Fourier transform
freqs = np.fft.fftfreq(len(time), d=time[1] - time[0])

# Define a Gaussian window
def apply_gaussian_window(fft_data, center_frequency, std_dev, time):
    N = len(fft_data)
    freqs = np.fft.fftfreq(N, d=(time[1] - time[0]))
    shifted_freqs = np.fft.fftshift(freqs)

    # Create the Gaussian window
    gaussian_window = np.exp(-0.5 * ((shifted_freqs - center_frequency)/ std_dev)**2)

    # Apply the Gaussian window to the FFT data
    fft_data_windowed = fft_data * gaussian_window

    return fft_data_windowed

# Function to filter harmonics
def filter_harmonics(J_omega, freqs, harmonics, omega_0):
    filtered_spectrum = np.zeros_like(J_omega)
    for n in harmonics:
        center_frequency = n * omega_0 /(2*np.pi)
        
        result = apply_gaussian_window(J_omega, center_frequency, std_dev, time)
        filtered_spectrum += result
    return filtered_spectrum

def calculate_fwhm(I_pulse, time):
    # Step 1: Find the index of the maximum peak
    peak_index = np.argmax(I_pulse)
    #print(time[peak_index]*0.6582)

    # Step 2: Calculate the half maximum
    half_max = I_pulse[peak_index] / 2.0

    # Step 3: Move left from the peak to find the crossing point
    left_index = peak_index
    while left_index > 0 and I_pulse[left_index] > half_max:
        left_index -= 1
    left_crossing = time[left_index]
    #print(left_crossing*0.6582)

    # Step 4: Move right from the peak to find the crossing point
    right_index = peak_index
    while right_index < len(I_pulse) - 1 and I_pulse[right_index] > half_max:
        right_index += 1
    right_crossing = time[right_index]
    #print(right_crossing*0.6582)
    # Step 5: Calculate the FWHM (difference in time)
    fwhm = right_crossing - left_crossing

    return fwhm * 658.2  # Return FWHM in the same time units as input

def is_isolated_pulse(I_pulse, time, threshold_ratio=0.7):
    peaks, _ = find_peaks(I_pulse)
    if len(peaks) == 0:
        return False
    
    # Find the height of the tallest peak
    max_peak_height = I_pulse[peaks].max()
    
    # Find all peaks that are above a certain threshold
    significant_peaks = I_pulse[peaks] > threshold_ratio * max_peak_height
    
    # If there's only one significant peak, it's considered isolated
    return significant_peaks.sum() == 1

#std_dev = 1  # Adjust this value to control the width of the Gaussian window

# Define the harmonic range you want to test
min_harmonic = 1  # Minimum harmonic to start with
max_harmonic = 31  # Maximum harmonic to end with

# Initialize variables to store the best result
shortest_pulse = np.inf
best_combination = None
best_I_pulse = None

# Sliding window approach for consecutive harmonics
for start_harmonic in range(min_harmonic, max_harmonic - 1):
    for width in range(2, max_harmonic - start_harmonic + 1):
        harmonics = list(range(start_harmonic, start_harmonic + width))

         # Filtered spectra for each component
        filtered_spectrum_x = filter_harmonics(resultx_shift, freqs, harmonics, omega_0)
        filtered_spectrum_y = filter_harmonics(resulty_shift, freqs, harmonics, omega_0)
        filtered_spectrum_z = filter_harmonics(resultz_shift, freqs, harmonics, omega_0)

        # Apply ifftshift before performing the inverse FFT
        filtered_spectrum_x = np.fft.ifftshift(filtered_spectrum_x)
        filtered_spectrum_y = np.fft.ifftshift(filtered_spectrum_y)
        filtered_spectrum_z = np.fft.ifftshift(filtered_spectrum_z)

        # Inverse Fourier Transform to get the electric field in each dimension
        E_pulse_x = np.fft.ifft(filtered_spectrum_x)
        E_pulse_y = np.fft.ifft(filtered_spectrum_y)
        E_pulse_z = np.fft.ifft(filtered_spectrum_z)

        # Combine the electric fields to form the total electric field
        E_total = np.sqrt(np.abs(E_pulse_x)**2 + np.abs(E_pulse_y)**2 + np.abs(E_pulse_z)**2)

        # Calculate the Attosecond Pulse Intensity
        I_pulse = np.abs(E_total) ** 2

        # Check if the pulse is isolated
        if is_isolated_pulse(I_pulse, time):
            # Calculate the pulse duration (FWHM)
            fwhm = calculate_fwhm(I_pulse, time)

            # Check if this is the shortest pulse
            if fwhm < shortest_pulse:
                shortest_pulse = fwhm
                best_combination = harmonics
                best_I_pulse = I_pulse


# Output the results
if best_combination is not None:
    print(f'Shortest isolated pulse duration: {shortest_pulse:.2f} as')
    print(f'Best harmonic combination: {best_combination}')
    '''   
    # Plot the best pulse
    plt.figure()
    plt.plot(time * 0.6582, best_I_pulse)
    plt.title('Isolated Attosecond Pulse Intensity')
    plt.xlabel('Time (fs)')
    plt.ylabel('Intensity (arb.)')
    plt.savefig('isolated_pulse.png')
    '''
    # Plot the best pulse
    plt.figure()
    plt.plot(time * 0.6582, best_I_pulse)
    plt.title('Isolated Attosecond Pulse Intensity')
    plt.xlabel('Time (fs)')
    plt.ylabel('Intensity (arb.)')

    dark_yellow = '#FFD700'
    
# Step 1: Find the half maximum of the peak
    half_max = np.max(best_I_pulse) / 2.0

# Step 2: Find indices where the signal crosses the half maximum
    left_index = np.where(best_I_pulse >= half_max)[0][0]
    right_index = np.where(best_I_pulse >= half_max)[0][-1]

    # Step 3: Calculate the FWHM (Full Width at Half Maximum)
    fwhm = (time[right_index] - time[left_index]) * 0.6582  # Convert to fs
    
    # Step 3: Add arrows -> <- to mark the FWHM (full width at half maximum)
    plt.annotate('', xy=(time[left_index]*0.6582*0.9, half_max), 
             xytext=(time[left_index]*0.6582*1.03, half_max),
             arrowprops=dict(arrowstyle='<-', color=dark_yellow))
    plt.annotate('', xy=(time[right_index]*0.6582*0.97, half_max), 
             xytext=(time[right_index]*0.6582*1.1, half_max),
             arrowprops=dict(arrowstyle='->', color=dark_yellow))

    # Step 5: Label the energy range on the plot
    plt.text(0.2, 0.9, '31 < E < 36 eV', horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=12, color='black')

    # Step 7: Annotate the FWHM on the plot
    plt.text(time[left_index]*0.6582*0.6, half_max*0.95, 
         f'FWHM = {shortest_pulse:.2f} as', horizontalalignment='center', color='black')


    #plt.legend()
    plt.savefig('isolated_pulse.png')
else:
    print('No isolated pulse found.')
    
# Output the results
print(f'Shortest pulse duration: {shortest_pulse:.2f} as')
print(f'Best harmonic combination: {best_combination}')

# Plot the best pulse
filtered_spectrum_x = filter_harmonics(resultx_shift, freqs, best_combination, omega_0)
filtered_spectrum_y = filter_harmonics(resulty_shift, freqs, best_combination, omega_0)
filtered_spectrum_z = filter_harmonics(resultz_shift, freqs, best_combination, omega_0)

# Apply ifftshift before performing the inverse FFT
filtered_spectrum_x = np.fft.ifftshift(filtered_spectrum_x)
filtered_spectrum_y = np.fft.ifftshift(filtered_spectrum_y)
filtered_spectrum_z = np.fft.ifftshift(filtered_spectrum_z)

E_pulse_x = np.fft.ifft(filtered_spectrum_x)
E_pulse_y = np.fft.ifft(filtered_spectrum_y)
E_pulse_z = np.fft.ifft(filtered_spectrum_z)

E_total = np.sqrt(np.abs(E_pulse_x)**2 + np.abs(E_pulse_y)**2 + np.abs(E_pulse_z)**2)
I_pulse = np.abs(E_total) ** 2

plt.figure()
plt.plot(time*0.6582, I_pulse)
plt.title('Attosecond Pulse Intensity')
plt.xlabel('Time (fs)')
plt.ylabel('Intensity (arb.)')
#plt.yscale('log')
plt.savefig('shortest_pulse.png')
