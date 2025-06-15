#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

// ‘ункц≥€ дл€ обчисленн€ визначника методом √ауса
double calculateDeterminant(std::vector<std::vector<double>>& matrix, int n) {
    double det = 1.0;

    for (int k = 0; k < n; ++k) {
        int maxRow = k;
        double maxElem = std::abs(matrix[k][k]);

        for (int i = k + 1; i < n; ++i) {
            if (std::abs(matrix[i][k]) > maxElem) {
                maxElem = std::abs(matrix[i][k]);
                maxRow = i;
            }
        }

        if (maxElem < 1e-9) return 0.0; // сингул€рна матриц€

        if (maxRow != k) {
            std::swap(matrix[k], matrix[maxRow]);
            det *= -1.0;
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

    return det;
}

int main() {
    int n, threads;

    std::cout << "Enter the size of the matrix (n x n): ";
    std::cin >> n;

    std::cout << "Enter the number of threads: ";
    std::cin >> threads;

    omp_set_num_threads(threads);

    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));

    srand(static_cast<unsigned>(time(nullptr)));

    for (auto& row : matrix)
        for (auto& val : row)
            val = static_cast<double>(rand() % 20 + 1);

    double start = omp_get_wtime();
    double det = calculateDeterminant(matrix, n);
    double end = omp_get_wtime();

    std::cout << "Determinant: " << det << "\n";
    std::cout << "Time elapsed: " << end - start << " seconds\n";

    return 0;
}
