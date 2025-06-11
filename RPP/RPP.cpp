#include <iostream>
#include <vector>
#include <iomanip> /
#include <omp.h>   
#include <cmath>   
#include <cstdlib>


void generate_matrix(std::vector<std::vector<double>>& matrix, int n) {
    srand(12345); 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = static_cast<double>(rand() % 1000) / 100.0 - 5.0; 
        }
    }
}


double calculate_determinant(std::vector<std::vector<double>>& matrix, int n) {
    double det = 1.0;
    int sign = 1;

    for (int k = 0; k < n; ++k) {
        
        int pivot_row = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(matrix[i][k]) > std::abs(matrix[pivot_row][k])) {
                pivot_row = i;
            }
        }

        if (std::abs(matrix[pivot_row][k]) < 1e-9) { 
            return 0.0;
        }

        
        if (pivot_row != k) {
            std::swap(matrix[k], matrix[pivot_row]);
            sign *= -1; 
        }

        det *= matrix[k][k]; 

        
#pragma omp parallel for
        for (int i = k + 1; i < n; ++i) {
            double factor = matrix[i][k] / matrix[k][k];
            for (int j = k; j < n; ++j) {
                matrix[i][j] -= factor * matrix[k][j];
            }
        }
    }

    return det * sign;
}

int main() {
    int n;
    int num_threads;

    std::cout << "Enter matrix size (N x N): ";
    std::cin >> n;

    std::cout << "Enter number of threads: ";
    std::cin >> num_threads;

    omp_set_num_threads(num_threads); 

    
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));

  
    double start_gen_time = omp_get_wtime();
    generate_matrix(matrix, n);
    double end_gen_time = omp_get_wtime();
    std::cerr << std::fixed << std::setprecision(6) << (end_gen_time - start_gen_time) << " s - finished data generation\n";

    
    double start_det_time = omp_get_wtime();
    double determinant = calculate_determinant(matrix, n);
    double end_det_time = omp_get_wtime();

    std::cout << "Determinant: " << std::fixed << std::setprecision(6) << determinant << std::endl;
    std::cerr << std::fixed << std::setprecision(6) << (end_det_time - start_det_time) << " s - finished computation\n";

    return 0;
}