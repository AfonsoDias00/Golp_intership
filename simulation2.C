#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <utility>

#include <gsl/gsl_integration.h>

// Function to read data from the text file
void readData(const std::string& filename, std::vector<double>& time, std::vector<double>& current, std::vector<double>& tau, std::vector<double>& energy) {
    std::ifstream file(filename);
    double t, c, t2, e;
    while (file >> t >> c) {
        time.push_back(t);
        current.push_back(c);
    }
    file.close();

    file.open(filename); // Reopen the file to read the remaining columns
    while (file >> t >> c >> t2 >> e) {
        tau.push_back(t2);
        energy.push_back(e);
    }
    file.close();

    std::cout << "oi" << std::endl;
}

std::vector<double> computeDerivative(const std::vector<double>& time, const std::vector<double>& current) {
    std::vector<double> derivative;
    derivative.reserve(current.size());

    // Use central difference for numerical differentiation
    for (size_t i = 1; i < current.size() - 1; ++i) {
        double dt = time[i + 1] - time[i - 1];
        double dc = current[i + 1] - current[i - 1];
        derivative.push_back(dc / dt);
    }

    // Use forward/backward difference for the endpoints
    derivative.push_back((current[1] - current[0]) / (time[1] - time[0]));
    derivative.push_back((current[current.size() - 1] - current[current.size() - 2]) / (time[time.size() - 1] - time[time.size() - 2]));

    return derivative;
}

// Define the integrand function for the complex integral
double integrand_real(double t, void* params) {

  /*
    double tau = *((double*)params); // Cast params to double pointer
    double energy = *((double*)params + 1); // Cast params to double pointer to access energy
    std::vector<double>& derivative = *((std::vector<double>*)((double*)params + 2)); // Cast params to vector pointer
  */

   std::vector<double>& tau = *((std::vector<double>*)((void**)params)[0]); // Extract tau vector
   std::vector<double>& energy = *((std::vector<double>*)((void**)params)[1]); // Extract energy vector
   std::vector<double>& derivative = *((std::vector<double>*)((void**)params)[2]); // Extract derivative vector

    double current_derivative = derivative[static_cast<int>(t / 0.007349)]; // Get the derivative value at time t

    // Compute the real and imaginary parts of the integrand separately
    double real_part = exp(-(t - tau[0]) * (t - tau[0])/(0.03125)) * cos(-t * energy[0]) * current_derivative;
    

    // Return the pair of real and imaginary parts
    return real_part;
}

double integrand_img(double t, void* params) {

 std::vector<double>& tau = *((std::vector<double>*)((void**)params)[0]); // Extract tau vector
  std::vector<double>& energy = *((std::vector<double>*)((void**)params)[1]); // Extract energy vector
  std::vector<double>& derivative = *((std::vector<double>*)((void**)params)[2]); // Extract derivative vector

  /*double tau = *((double*)params); // Cast params to double pointer
    double energy = *((double*)params + 1); // Cast params to double pointer to access energy
    std::vector<double>& derivative = *((std::vector<double>*)((double*)params + 2)); // Cast params to vector pointer
  */
    double current_derivative = derivative[static_cast<int>(t / 0.007349)]; // Get the derivative value at time t

    
    double imag_part = exp(-(t - tau[0]) * (t - tau[0])/(0.03125)) * sin(-t * energy[0]) * current_derivative;

    // Return the pair of real and imaginary parts
    return imag_part;
}

// Function to perform complex integration for a given tau and energy
double performComplexIntegral(const std::vector<double>& time, double tau, double energy, const std::vector<double>& derivative) {
  //   gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(18451);
    gsl_function F_real, F_imag;
    F_real.function = &integrand_real;
    F_imag.function = &integrand_img;
    std::vector<double> tau_v = {tau};
    std::vector<double> energy_v = {energy};
    const std::vector<double>* params[3] = {&tau_v, &energy_v, &derivative}; // Parameters for the integrand function
    F_real.params = params;
    F_imag.params = params;

    double result_real, result_imag, abserr_real, abserr_imag;
    double abs_error = 1.0e-1;
    
    gsl_integration_qng(&F_real, 0, 89.255, 0, abs_error, &result_real, &abserr_real, nullptr);
    gsl_integration_qng(&F_imag, 0, 89.255, 0, abs_error, &result_imag, &abserr_imag, nullptr); // Integrate imag part from 0 to infinity

    //    gsl_integration_workspace_free(workspace);

    double norm = std::sqrt(result_real * result_real + result_imag * result_imag);
    return norm;
}

void writeData(const std::vector<std::vector<double>>& results, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i != row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
}

int main() {
    std::vector<double> time, current, tau, energy;
    readData("eixos.txt", time, current, tau, energy);

    // Compute the derivative of current with respect to time
   
// Compute the derivative of current with respect to time
    std::vector<double> current_derivative = computeDerivative(time, current);

    std::vector<std::vector<double>> results(tau.size(), std::vector<double>(energy.size())); // Matrix to store the calculated results
    std::cout << "oi" << std::endl;
    // Loop over tau and energy to calculate the 2D function
    for (size_t i = 0; i < tau.size(); ++i) {
        for (size_t j = 0; j < energy.size(); ++j) {
            double result = performComplexIntegral(time, tau[i], energy[j], current_derivative);
            results[i][j] = result; // Store the result in the matrix
        }
    }
    std::cout << "oi" << std::endl;
    // Write the results matrix to a CSV file
    writeData(results, "results.csv");

    return 0;
}
