#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SINGULAR_THRESHOLD 1e-12
typedef struct {
    int n;
    double **a;
} Matrix;
static Matrix alloc_matrix(int n) {
    Matrix m = {n, (double **)malloc(n * sizeof(double *))};
    if (!m.a) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    for (int i = 0; i < n; i++) {
        m.a[i] = (double *)malloc((n + 1) * sizeof(double));
        if (!m.a[i]) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    }
    return m;
}
static void free_matrix(Matrix *m) {
    for (int i = 0; i < m->n; i++) free(m->a[i]);
    free(m->a);
    m->a = NULL;
}
static void swap_rows(double **a, int cols, int r1, int r2) {
    double *tmp = a[r1];
    a[r1] = a[r2];
    a[r2] = tmp;
}
static int forward_eliminate(Matrix *m) {
    int n = m->n;
    double **a = m->a;
    for (int k = 0; k < n; k++) {
        int max_row = k;
        for (int i = k + 1; i < n; i++)
            if (fabs(a[i][k]) > fabs(a[max_row][k]))
                max_row = i;
        if (fabs(a[max_row][k]) < SINGULAR_THRESHOLD)
            return -1;
        if (max_row != k)
            swap_rows(a, n + 1, k, max_row);
        for (int i = k + 1; i < n; i++) {
            double factor = a[i][k] / a[k][k];
            for (int j = k; j <= n; j++)
                a[i][j] -= factor * a[k][j];
            a[i][k] = 0.0;
        }
    }
    return 0;
}
static void back_substitute(Matrix *m, double *x) {
    int n = m->n;
    double **a = m->a;
    for (int i = n - 1; i >= 0; i--) {
        x[i] = a[i][n];
        for (int j = i + 1; j < n; j++)
            x[i] -= a[i][j] * x[j];
        x[i] /= a[i][i];
    }
}
static double compute_residual(Matrix *orig, double *x) {
    double max_res = 0.0;
    int n = orig->n;
    for (int i = 0; i < n; i++) {
        double r = orig->a[i][n];
        for (int j = 0; j < n; j++)
            r -= orig->a[i][j] * x[j];
        if (fabs(r) > max_res) max_res = fabs(r);
    }
    return max_res;
}
static int read_int(const char *prompt, int min, int max) {
    int val;
    printf("%s", prompt);
    while (scanf("%d", &val) != 1 || val < min || val > max) {
        fprintf(stderr, "Enter an integer between %d and %d.\n", min, max);
        while (getchar() != '\n');
        printf("%s", prompt);
    }
    return val;
}
static void read_matrix(Matrix *m) {
    printf("Enter augmented matrix (%d x %d) row by row:\n", m->n, m->n + 1);
    for (int i = 0; i < m->n; i++) {
        printf("  Row %d: ", i + 1);
        for (int j = 0; j <= m->n; j++) {
            while (scanf("%lf", &m->a[i][j]) != 1) {
                fprintf(stderr, "Invalid input, try again: ");
                while (getchar() != '\n');
            }
        }
    }
}
int main(void) {
    int n = read_int("How many variables/equations? ", 1, 1000);
    Matrix working = alloc_matrix(n);
    Matrix original = alloc_matrix(n);
    read_matrix(&working);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= n; j++)
            original.a[i][j] = working.a[i][j];
    if (forward_eliminate(&working) != 0) {
        printf("No unique solution (singular or near-singular matrix).\n");
        free_matrix(&working);
        free_matrix(&original);
        return EXIT_FAILURE;
    }
    double *x = (double *)malloc(n * sizeof(double));
    if (!x) { fprintf(stderr, "malloc failed\n"); return EXIT_FAILURE; }
    back_substitute(&working, x);
    printf("\nSolution:\n");
    for (int i = 0; i < n; i++)
        printf("  x%-4d = %+.10g\n", i + 1, x[i]);
    printf("\nResidual ||Ax - b||_inf = %.6e\n", compute_residual(&original, x));
    free(x);
    free_matrix(&working);
    free_matrix(&original);
    return EXIT_SUCCESS;
}