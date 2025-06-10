#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

double determinant(vector<vector<double>> matrix, int n) {
    if (n == 1) return matrix[0][0];
    if (n == 2)
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

    double det = 0.0;

#pragma omp parallel for reduction(+:det)
    for (int p = 0; p < n; ++p) {
        vector<vector<double>> submatrix(n - 1, vector<double>(n - 1));
        for (int i = 1; i < n; ++i) {
            int colIdx = 0;
            for (int j = 0; j < n; ++j) {
                if (j == p) continue;
                submatrix[i - 1][colIdx++] = matrix[i][j];
            }
        }
        double sign = (p % 2 == 0) ? 1 : -1;
        det += sign * matrix[0][p] * determinant(submatrix, n - 1);
    }
    return det;
}

int main() {
    int n;
    cout << "Enter matrix size (n x n): ";
    cin >> n;
    vector<vector<double>> matrix(n, vector<double>(n));

    cout << "Enter matrix values row by row:\n";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> matrix[i][j];

    double start = omp_get_wtime();
    double result = determinant(matrix, n);
    double end = omp_get_wtime();

    cout << "Determinant: " << result << endl;
    cout << "Time: " << (end - start) << " seconds" << endl;
    return 0;
}
