#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

double MatrixDeterminant(int nDim, std::vector<double>& matrix) {
    double det = 1.0;
    for (int k = 0; k < nDim - 1; ++k) {
        double maxElem = std::abs(matrix[k * nDim + k]);
        int maxRow = k;
        for (int i = k + 1; i < nDim; ++i) {
            if (std::abs(matrix[i * nDim + k]) > maxElem) {
                maxElem = std::abs(matrix[i * nDim + k]);
                maxRow = i;
            }
        }

        if (maxRow != k) {
            for (int i = k; i < nDim; ++i)
                std::swap(matrix[k * nDim + i], matrix[maxRow * nDim + i]);
            det *= -1.0;
        }

        if (matrix[k * nDim + k] == 0.0) return 0.0;

#pragma omp parallel for
        for (int j = k + 1; j < nDim; ++j) {
            double factor = -matrix[j * nDim + k] / matrix[k * nDim + k];
            for (int i = k; i < nDim; ++i) {
                matrix[j * nDim + i] += factor * matrix[k * nDim + i];
            }
        }
    }

    for (int i = 0; i < nDim; ++i)
        det *= matrix[i * nDim + i];

    return det;
}

int main() {
    int n;
    std::cout << "Enter the size of the matrix (n x n): ";
    std::cin >> n;

    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    srand(static_cast<unsigned>(time(nullptr)));

    for (auto& row : matrix)
        for (auto& val : row)
            val = static_cast<double>(rand() % 20 + 1);

    std::vector<double> flatMatrix(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flatMatrix[i * n + j] = matrix[i][j];

    double startTime = omp_get_wtime();
    double determinant = MatrixDeterminant(n, flatMatrix);
    double endTime = omp_get_wtime();

    std::cout << "Determinant: " << determinant << std::endl;
    std::cout << "Time elapsed: " << endTime - startTime << " seconds" << std::endl;

    return 0;
}
