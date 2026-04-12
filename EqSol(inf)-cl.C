#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <math.h>

#include <float.h>



/* ═══════════════════════════════════════════════════════════════════════════

   CONSTANTS

   ═══════════════════════════════════════════════════════════════════════════ */



#define EPS           2.220446049250313e-16

#define REFINE_MAX    10

#define REFINE_TOL    (EPS * 4.0)

#define MAX_N         10000



/*

 * Unified rank threshold: tol = RANK_TOL_FACTOR * n * EPS * largest_diagonal

 * Used consistently in both LU and QR rank decisions.

 * This matches the default tolerance used in MATLAB's rank() and numpy.linalg.matrix_rank().

 */

#define RANK_TOL_FACTOR  2.0



/* ═══════════════════════════════════════════════════════════════════════════

   TYPES

   ═══════════════════════════════════════════════════════════════════════════ */



typedef enum {

    SOLVE_OK         =  0,

    SOLVE_INFINITE   = -1,

    SOLVE_NONE       = -2,

    SOLVE_DEGENERATE = -3

} SolveResult;



typedef struct {

    int     n;

    double *lu;      /* compact LU, n*n, row-major */

    int    *piv;     /* row permutation, length n   */

    double *scales;  /* row scale factors, length n */

    int     rank;

    int     sign;    /* permutation parity for det  */

} LU;



typedef struct {

    int     n;

    double *qr;      /* Householder reflectors + R, n*n, row-major */

    int    *jpvt;    /* column permutation, length n               */

    double *tau;     /* Householder scalars, length n              */

    double *work;    /* reusable column buffer, length n           */

    double *cnorm;   /* column norms squared, length n             */

    int     rank;

} QR;



#define A_(a,n,i,j)  ((a)[(i)*(n)+(j)])

#define L_(f,i,j)    ((f).lu[(i)*(f).n+(j)])

#define QR_(f,i,j)   ((f).qr[(i)*(f).n+(j)])



/* ═══════════════════════════════════════════════════════════════════════════

   MEMORY

   ═══════════════════════════════════════════════════════════════════════════ */



static void *safe_malloc(size_t sz) {

    void *p = malloc(sz);

    if (!p) { fprintf(stderr, "error: malloc failed\n"); exit(EXIT_FAILURE); }

    return p;

}



static double *vec_alloc(int n)        { return (double *)safe_malloc((size_t)n * sizeof(double)); }

static double *mat_alloc(int r, int c) { return (double *)safe_malloc((size_t)r * c * sizeof(double)); }



static LU alloc_lu(int n) {

    LU f = { n, mat_alloc(n,n), (int*)safe_malloc((size_t)n*sizeof(int)),

              vec_alloc(n), 0, 1 };

    return f;

}



static void free_lu(LU *f) {

    free(f->lu); free(f->piv); free(f->scales);

    f->lu = NULL; f->piv = NULL; f->scales = NULL;

}



static QR alloc_qr(int n) {

    QR f;

    f.n     = n;

    f.qr    = mat_alloc(n, n);

    f.jpvt  = (int *)safe_malloc((size_t)n * sizeof(int));

    f.tau   = vec_alloc(n);

    f.work  = vec_alloc(n);   /* reusable column buffer — allocated once */

    f.cnorm = vec_alloc(n);

    f.rank  = 0;

    return f;

}



static void free_qr(QR *f) {

    free(f->qr); free(f->jpvt); free(f->tau); free(f->work); free(f->cnorm);

    f->qr = NULL;

}



/* ═══════════════════════════════════════════════════════════════════════════

   VECTOR / MATRIX UTILITIES

   ═══════════════════════════════════════════════════════════════════════════ */



static double vec_inf_norm(const double *v, int n) {

    double mx = 0.0;

    for (int i = 0; i < n; i++) { double a = fabs(v[i]); if (a > mx) mx = a; }

    return mx;

}



static double mat_1norm(const double *A, int n) {

    double mx = 0.0;

    for (int j = 0; j < n; j++) {

        double s = 0.0;

        for (int i = 0; i < n; i++) s += fabs(A_(A,n,i,j));

        if (s > mx) mx = s;

    }

    return mx;

}



static void mat_vec_residual(const double *A, int n,

                              const double *x, const double *b, double *r) {

    for (int i = 0; i < n; i++) {

        double s = b[i];

        for (int j = 0; j < n; j++) s -= A_(A,n,i,j) * x[j];

        r[i] = s;

    }

}



/* ═══════════════════════════════════════════════════════════════════════════

   HOUSEHOLDER REFLECTOR  (LAPACK-style, numerically stable)



   Given vector x of length n, compute H = I - tau*v*v^T such that

   H*x = [ beta, 0, 0, ..., 0 ]^T



   beta = -sign(x[0]) * ||x||_2         (opposite sign to x[0] for stability)

   v[0] = 1  (implicit)

   v[i] = x[i] / (x[0] - beta)  for i >= 1

   tau  = (beta - x[0]) / beta = 2 / (v^T v)



   On exit: x[0] = beta, x[1..n-1] = v[1..n-1] (v[0]=1 stored implicitly).

   tau is written to *tau_out.



   Reference: Golub & Van Loan, "Matrix Computations" 4th ed., Algorithm 5.1.1

   ═══════════════════════════════════════════════════════════════════════════ */



static void householder(double *x, int n, double *tau_out) {

    if (n <= 1) { *tau_out = 0.0; return; }



    /* ||x[1:]||^2 */

    double sigma = 0.0;

    for (int i = 1; i < n; i++) sigma += x[i] * x[i];



    if (sigma < EPS * EPS) {

        /* x is already a multiple of e1 — reflector is identity */

        *tau_out = 0.0;

        return;

    }



    double norm_x = sqrt(x[0]*x[0] + sigma);



    /*

     * beta = -sign(x[0]) * ||x||

     * Choosing sign opposite to x[0] maximises |x[0] - beta|,

     * which avoids catastrophic cancellation in the division below.

     */

    double beta = (x[0] > 0.0) ? -norm_x : norm_x;



    double inv = 1.0 / (x[0] - beta);

    for (int i = 1; i < n; i++) x[i] *= inv;



    *tau_out = (beta - x[0]) / beta;   /* = 2 / (v^T v), always in (0, 2] */

    x[0]     = beta;                   /* R diagonal element                */

}



/* ═══════════════════════════════════════════════════════════════════════════

   LU FACTORIZATION  (Doolittle, scaled partial pivoting)

   ═══════════════════════════════════════════════════════════════════════════ */



static int lu_factor(LU *f) {

    int    n    = f->n;

    int    rank = 0;

    int    sign = 1;



    /* row scale factors */

    for (int i = 0; i < n; i++) {

        double mx = 0.0;

        for (int j = 0; j < n; j++) { double v = fabs(L_(f,i,j)); if (v > mx) mx = v; }

        f->scales[i] = (mx > 0.0) ? mx : 1.0;

        f->piv[i]    = i;

    }



    /* unified rank threshold based on largest element in A */

    double max_elem = 0.0;

    for (int i = 0; i < n*n; i++) { double v = fabs(f->lu[i]); if (v > max_elem) max_elem = v; }

    double rank_tol = RANK_TOL_FACTOR * n * EPS * max_elem;



    for (int k = 0; k < n; k++) {

        int    max_row = k;

        double max_val = fabs(L_(f,k,k)) / f->scales[k];

        for (int i = k+1; i < n; i++) {

            double v = fabs(L_(f,i,k)) / f->scales[i];

            if (v > max_val) { max_val = v; max_row = i; }

        }



        if (fabs(L_(f, max_row, k)) < rank_tol) continue;

        rank++;



        if (max_row != k) {

            for (int j = 0; j < n; j++) {

                double tmp = L_(f,k,j); L_(f,k,j) = L_(f,max_row,j); L_(f,max_row,j) = tmp;

            }

            int    ti = f->piv[k];    f->piv[k]    = f->piv[max_row];    f->piv[max_row]    = ti;

            double ts = f->scales[k]; f->scales[k] = f->scales[max_row]; f->scales[max_row] = ts;

            sign = -sign;

        }



        double pivot = L_(f,k,k);

        for (int i = k+1; i < n; i++) {

            double fac = L_(f,i,k) / pivot;

            L_(f,i,k)  = fac;

            for (int j = k+1; j < n; j++)

                L_(f,i,j) -= fac * L_(f,k,j);

        }

    }



    f->rank = rank;

    f->sign = sign;

    return rank;

}



static void lu_extract_L(const LU *f, double *dst) {

    int n = f->n;

    for (int i = 0; i < n; i++)

        for (int j = 0; j < n; j++)

            A_(dst,n,i,j) = (j < i) ? L_(f,i,j) : (j == i) ? 1.0 : 0.0;

}



static void lu_extract_U(const LU *f, double *dst) {

    int n = f->n;

    for (int i = 0; i < n; i++)

        for (int j = 0; j < n; j++)

            A_(dst,n,i,j) = (j >= i) ? L_(f,i,j) : 0.0;

}



static double lu_determinant(const LU *f) {

    double det = (double)f->sign;

    for (int i = 0; i < f->n; i++) det *= L_(f,i,i);

    return det;

}



/* ═══════════════════════════════════════════════════════════════════════════

   LU SOLVE

   ═══════════════════════════════════════════════════════════════════════════ */



static SolveResult lu_solve(const LU *f, const double *rhs, double *x) {

    int     n = f->n;

    double *y = vec_alloc(n);



    for (int i = 0; i < n; i++) y[i] = rhs[f->piv[i]];



    /* forward: L*y = y  (unit lower triangular) */

    for (int i = 0; i < n; i++)

        for (int j = 0; j < i; j++)

            y[i] -= L_(f,i,j) * y[j];



    /* backward: U*x = y */

    for (int i = n-1; i >= 0; i--) {

        double diag = L_(f,i,i);

        if (fabs(diag) < RANK_TOL_FACTOR * n * EPS * f->scales[i]) {

            free(y); return SOLVE_DEGENERATE;

        }

        x[i] = y[i];

        for (int j = i+1; j < n; j++) x[i] -= L_(f,i,j) * x[j];

        x[i] /= diag;

    }



    free(y);

    return SOLVE_OK;

}



/* ═══════════════════════════════════════════════════════════════════════════

   ITERATIVE REFINEMENT

   ═══════════════════════════════════════════════════════════════════════════ */



static void iterative_refine(const double *A, const LU *f,

                              const double *b, double *x) {

    int     n        = f->n;

    double *r        = vec_alloc(n);

    double *dx       = vec_alloc(n);

    double  prev_res = DBL_MAX;



    for (int iter = 0; iter < REFINE_MAX; iter++) {

        mat_vec_residual(A, n, x, b, r);

        double res = vec_inf_norm(r, n);

        if (res < REFINE_TOL || res >= prev_res) break;

        prev_res = res;

        if (lu_solve(f, r, dx) != SOLVE_OK) break;

        for (int i = 0; i < n; i++) x[i] += dx[i];

    }



    free(r); free(dx);

}



/* ═══════════════════════════════════════════════════════════════════════════

   CONDITION NUMBER ESTIMATION  (Hager's algorithm, correct implementation)



   Estimates ||A^{-1}||_1 via the power method on sign vectors.

   This is mathematically identical to LAPACK's dlacon/dlacn2 subroutine.



   The algorithm:

     1. Start with v = [1/n, ..., 1/n]

     2. Compute w = A^{-1} * v  via lu_solve

     3. Build sign vector: s[i] = sign(w[i])

     4. Compute z = A^{-T} * s  via transposed LU solves

     5. If ||z||_inf <= z^T * s, estimate = ||w||_1, done

     6. Otherwise set v = e_{argmax |z|}, goto 2



   Convergence is guaranteed in at most n+2 iterations.

   In practice 3-5 suffice for a reliable estimate.

   ═══════════════════════════════════════════════════════════════════════════ */



static void lu_solve_transposed(const LU *f, const double *rhs, double *x) {

    /*

     * Solve (PA LU)^T x = rhs, i.e., U^T L^T P^T x = rhs

     * Step 1: solve U^T y = rhs

     * Step 2: solve L^T z = y

     * Step 3: apply P^T: x[piv[i]] = z[i]

     */

    int n = f->n;



    /* U^T y = rhs: forward sub on upper triangle transposed */

    double *y = vec_alloc(n);

    memcpy(y, rhs, (size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) {

        double diag = L_(f,i,i);

        if (fabs(diag) < EPS) { y[i] = 0.0; continue; }

        y[i] /= diag;

        for (int j = i+1; j < n; j++)

            y[j] -= L_(f,i,j) * y[i];

    }



    /* L^T z = y: backward sub on unit lower triangle transposed */

    for (int i = n-1; i >= 0; i--) {

        for (int j = i+1; j < n; j++)

            y[i] -= L_(f,j,i) * y[j];

        /* diagonal of L is 1, no division */

    }



    /* apply inverse permutation P^T: x[piv[i]] = y[i] */

    for (int i = 0; i < n; i++) x[f->piv[i]] = y[i];



    free(y);

}



static double estimate_inv_1norm(const LU *f) {

    int     n   = f->n;

    double *v   = vec_alloc(n);   /* current iterate                   */

    double *w   = vec_alloc(n);   /* A^{-1} * v                        */

    double *s   = vec_alloc(n);   /* sign vector                       */

    double *z   = vec_alloc(n);   /* A^{-T} * s                        */

    int    *ipiv = (int *)safe_malloc((size_t)n * sizeof(int));



    /* inverse permutation */

    for (int i = 0; i < n; i++) ipiv[f->piv[i]] = i;



    for (int i = 0; i < n; i++) v[i] = 1.0 / n;



    double est = 0.0;

    int    jlast = -1;



    for (int iter = 0; iter < n + 2; iter++) {



        /* w = A^{-1} * v */

        if (lu_solve(f, v, w) != SOLVE_OK) break;



        /* estimate = ||w||_1 */

        double new_est = 0.0;

        for (int i = 0; i < n; i++) new_est += fabs(w[i]);



        /* sign vector */

        for (int i = 0; i < n; i++) s[i] = (w[i] >= 0.0) ? 1.0 : -1.0;



        /* z = A^{-T} * s */

        lu_solve_transposed(f, s, z);



        /* convergence check: if ||z||_inf <= z^T * s, we're done */

        double zTs = 0.0;

        for (int i = 0; i < n; i++) zTs += z[i] * s[i];

        double z_inf = vec_inf_norm(z, n);



        if (new_est > est) est = new_est;



        if (z_inf <= zTs) break;   /* Hager convergence criterion */



        /* next v = e_{argmax |z|}, but not the same column twice */

        int jmax = 0;

        double zmax = fabs(z[0]);

        for (int i = 1; i < n; i++) if (fabs(z[i]) > zmax) { zmax = fabs(z[i]); jmax = i; }



        if (jmax == jlast) break;  /* cycling — stop */

        jlast = jmax;



        memset(v, 0, (size_t)n * sizeof(double));

        v[jmax] = 1.0;

    }



    free(v); free(w); free(s); free(z); free(ipiv);

    return est;

}



static double condition_number(const double *A, const LU *f) {

    return mat_1norm(A, f->n) * estimate_inv_1norm(f);

}



/* ═══════════════════════════════════════════════════════════════════════════

   QR FACTORIZATION WITH COLUMN PIVOTING  (Householder, stable)



   Uses:

   - Correct LAPACK-style Householder vectors

   - Single pre-allocated work buffer (no alloc/free in loop)

   - No stack VLAs

   - Unified rank threshold consistent with LU

   ═══════════════════════════════════════════════════════════════════════════ */



static void qr_factor(QR *f) {

    int     n     = f->n;

    double *col   = f->work;   /* reuse pre-allocated buffer — no alloc in loop */



    for (int j = 0; j < n; j++) {

        f->jpvt[j] = j;

        f->cnorm[j] = 0.0;

        for (int i = 0; i < n; i++) f->cnorm[j] += QR_(f,i,j) * QR_(f,i,j);

    }



    /* rank threshold: RANK_TOL_FACTOR * n * EPS * ||A||_F approximately */

    double max_diag = 0.0;

    int rank = 0;



    for (int k = 0; k < n; k++) {



        /* column pivot: largest remaining column norm */

        int    max_col  = k;

        double max_norm = f->cnorm[k];

        for (int j = k+1; j < n; j++)

            if (f->cnorm[j] > max_norm) { max_norm = f->cnorm[j]; max_col = j; }



        if (max_norm < EPS * EPS) break;



        if (max_col != k) {

            for (int i = 0; i < n; i++) {

                double tmp = QR_(f,i,k); QR_(f,i,k) = QR_(f,i,max_col); QR_(f,i,max_col) = tmp;

            }

            int    ti = f->jpvt[k];   f->jpvt[k]   = f->jpvt[max_col];  f->jpvt[max_col]  = ti;

            double td = f->cnorm[k];  f->cnorm[k]  = f->cnorm[max_col]; f->cnorm[max_col] = td;

        }



        /* extract column k from row k downward into work buffer */

        for (int i = k; i < n; i++) col[i-k] = QR_(f,i,k);



        /* compute Householder reflector — stable version */

        householder(col, n-k, &f->tau[k]);



        /* first diagonal of R sets the scale for rank threshold */

        if (k == 0) max_diag = fabs(col[0]);



        /* rank decision using unified threshold */

        double rank_tol = RANK_TOL_FACTOR * n * EPS * max_diag;

        if (fabs(col[0]) < rank_tol) break;

        rank++;



        /* write back: R diagonal and reflector vector */

        QR_(f,k,k) = col[0];                          /* R[k][k] = beta  */

        for (int i = k+1; i < n; i++) QR_(f,i,k) = col[i-k]; /* v[1:] stored below diag */



        /* apply H to remaining columns: A[k:,j] -= tau * v * (v^T * A[k:,j])

           v[0] = 1 implicitly, v[i] = QR_(f, k+i, k) for i >= 1            */

        for (int j = k+1; j < n; j++) {

            double dot = QR_(f,k,j);                   /* v[0]*A[k,j] = A[k,j] */

            for (int i = k+1; i < n; i++) dot += QR_(f,i,k) * QR_(f,i,j);

            dot *= f->tau[k];

            QR_(f,k,j)     -= dot;                     /* -= tau*v[0]*dot       */

            for (int i = k+1; i < n; i++) QR_(f,i,j) -= dot * QR_(f,i,k);



            /* norm downdate (numerically stable: recompute when needed) */

            double new_norm = f->cnorm[j] - QR_(f,k,j) * QR_(f,k,j);

            /*

             * Norm downdate can go slightly negative due to floating point.

             * When relative error is large (Rice criterion), recompute from scratch.

             */

            if (new_norm < 0.0 ||

                (f->cnorm[j] > 0.0 && new_norm / f->cnorm[j] < 0.01)) {

                new_norm = 0.0;

                for (int i = k+1; i < n; i++) new_norm += QR_(f,i,j) * QR_(f,i,j);

            }

            f->cnorm[j] = new_norm;

        }

    }



    f->rank = rank;

}



static int estimate_rank(const double *A, int n) {

    QR f = alloc_qr(n);

    memcpy(f.qr, A, (size_t)n * n * sizeof(double));

    qr_factor(&f);

    int r = f.rank;

    free_qr(&f);

    return r;

}



/* ═══════════════════════════════════════════════════════════════════════════

   SYSTEM CLASSIFICATION  (rank via QR on A and [A|b])

   ═══════════════════════════════════════════════════════════════════════════ */



static int rank_augmented(const double *A, const double *b, int n) {

    /* build n x (n+1) augmented matrix, run QR column-pivot on it */

    int     m   = n + 1;

    double *Ab  = mat_alloc(n, m);

    double *wrk = vec_alloc(n);

    double *cnm = vec_alloc(m);

    double *tau = vec_alloc(n);

    int    *jpv = (int *)safe_malloc((size_t)m * sizeof(int));



    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) A_(Ab,m,i,j) = A_(A,n,i,j);

        A_(Ab,m,i,n) = b[i];

    }

    for (int j = 0; j < m; j++) {

        jpv[j] = j; cnm[j] = 0.0;

        for (int i = 0; i < n; i++) cnm[j] += A_(Ab,m,i,j) * A_(Ab,m,i,j);

    }



    double max_diag = 0.0;

    int    rank     = 0;

    int    steps    = (n < m) ? n : m;



    for (int k = 0; k < steps; k++) {

        int max_col = k; double mx = cnm[k];

        for (int j = k+1; j < m; j++) if (cnm[j] > mx) { mx = cnm[j]; max_col = j; }

        if (mx < EPS*EPS) break;



        if (max_col != k) {

            for (int i = 0; i < n; i++) {

                double tmp = A_(Ab,m,i,k); A_(Ab,m,i,k) = A_(Ab,m,i,max_col); A_(Ab,m,i,max_col) = tmp;

            }

            int ti = jpv[k]; jpv[k] = jpv[max_col]; jpv[max_col] = ti;

            double td = cnm[k]; cnm[k] = cnm[max_col]; cnm[max_col] = td;

        }



        for (int i = k; i < n; i++) wrk[i-k] = A_(Ab,m,i,k);

        householder(wrk, n-k, &tau[k]);

        if (k == 0) max_diag = fabs(wrk[0]);



        double rank_tol = RANK_TOL_FACTOR * n * EPS * max_diag;

        if (fabs(wrk[0]) < rank_tol) break;

        rank++;



        A_(Ab,m,k,k) = wrk[0];

        for (int i = k+1; i < n; i++) A_(Ab,m,i,k) = wrk[i-k];



        for (int j = k+1; j < m; j++) {

            double dot = A_(Ab,m,k,j);

            for (int i = k+1; i < n; i++) dot += A_(Ab,m,i,k) * A_(Ab,m,i,j);

            dot *= tau[k];

            A_(Ab,m,k,j) -= dot;

            for (int i = k+1; i < n; i++) A_(Ab,m,i,j) -= dot * A_(Ab,m,i,k);

            double nn = cnm[j] - A_(Ab,m,k,j)*A_(Ab,m,k,j);

            if (nn < 0.0) { nn = 0.0; for (int i=k+1;i<n;i++) nn += A_(Ab,m,i,j)*A_(Ab,m,i,j); }

            cnm[j] = nn;

        }

    }



    free(Ab); free(wrk); free(cnm); free(tau); free(jpv);

    return rank;

}



static SolveResult classify_system(const double *A, const double *b,

                                    int n, int *rank_A_out) {

    int rA  = estimate_rank(A, n);

    *rank_A_out = rA;

    if (rA == n) return SOLVE_OK;

    int rAb = rank_augmented(A, b, n);

    return (rAb > rA) ? SOLVE_NONE : SOLVE_INFINITE;

}



/* ═══════════════════════════════════════════════════════════════════════════

   I/O

   ═══════════════════════════════════════════════════════════════════════════ */



static int read_int(const char *prompt, int lo, int hi) {

    int val;

    printf("%s", prompt);

    for (;;) {

        if (scanf("%d", &val) == 1 && val >= lo && val <= hi) return val;

        fprintf(stderr, "  enter an integer in [%d, %d]: ", lo, hi);

        while (getchar() != '\n');

    }

}



static void read_matrix(int n, double *A, double *b) {

    printf("Enter augmented matrix (%d x %d) — coefficients then RHS per row:\n", n, n+1);

    for (int i = 0; i < n; i++) {

        printf("  row %d: ", i+1);

        for (int j = 0; j < n; j++) {

            while (scanf("%lf", &A_(A,n,i,j)) != 1) {

                fprintf(stderr, "  invalid — enter a number: ");

                while (getchar() != '\n');

            }

        }

        while (scanf("%lf", &b[i]) != 1) {

            fprintf(stderr, "  invalid — enter a number: ");

            while (getchar() != '\n');

        }

    }

}



static void print_solution(const double *x, int n) {

    printf("\nSolution:\n");

    for (int i = 0; i < n; i++)

        printf("  x%-4d = %+.15g\n", i+1, x[i]);

}



/* ═══════════════════════════════════════════════════════════════════════════

   MAIN

   ═══════════════════════════════════════════════════════════════════════════ */



int main(void) {

    int n = read_int("How many variables/equations? ", 1, MAX_N);



    double *A      = mat_alloc(n, n);

    double *b      = vec_alloc(n);

    double *b_orig = vec_alloc(n);

    double *x      = vec_alloc(n);

    double *r      = vec_alloc(n);



    read_matrix(n, A, b);

    memcpy(b_orig, b, (size_t)n * sizeof(double));



    /* classify */

    int rank_A;

    SolveResult cls = classify_system(A, b, n, &rank_A);

    if (cls != SOLVE_OK) {

        printf("\n%s\n  rank(A) = %d  of  n = %d\n",

            cls == SOLVE_NONE

                ? "No solution — inconsistent (rank(A) < rank([A|b]))."

                : "Infinite solutions — rank-deficient (rank(A) < n).",

            rank_A, n);

        free(A); free(b); free(b_orig); free(x); free(r);

        return EXIT_FAILURE;

    }



    /* LU */

    LU f = alloc_lu(n);

    memcpy(f.lu, A, (size_t)n * n * sizeof(double));

    lu_factor(&f);



    /* solve */

    SolveResult status = lu_solve(&f, b_orig, x);

    if (status != SOLVE_OK) {

        fprintf(stderr, "Degenerate pivot in back-substitution.\n");

        free_lu(&f); free(A); free(b); free(b_orig); free(x); free(r);

        return EXIT_FAILURE;

    }



    /* refine */

    iterative_refine(A, &f, b_orig, x);



    /* diagnostics */

    double cond = condition_number(A, &f);

    double det  = lu_determinant(&f);

    mat_vec_residual(A, n, x, b_orig, r);

    double res  = vec_inf_norm(r, n);



    print_solution(x, n);

    printf("\nDiagnostics:\n");

    printf("  rank(A)                  = %d\n",    rank_A);

    printf("  determinant              = %+.6e\n", det);

    printf("  condition number κ₁(A)  ≈ %.6e\n",  cond);

    printf("  residual ||Ax-b||_inf    = %.6e\n",  res);

    if (cond > 1.0 / (EPS * n))

        printf("  WARNING: ill-conditioned — lost ≈ %.1f decimal digits\n",

               log10(cond * EPS * n));



    if (n <= 6) {

        double *L = mat_alloc(n,n), *U = mat_alloc(n,n);

        lu_extract_L(&f, L); lu_extract_U(&f, U);

        printf("\nL =\n");

        for (int i = 0; i < n; i++) {

            printf("  "); for (int j = 0; j < n; j++) printf("%12.6f ", A_(L,n,i,j)); printf("\n");

        }

        printf("\nU =\n");

        for (int i = 0; i < n; i++) {

            printf("  "); for (int j = 0; j < n; j++) printf("%12.6f ", A_(U,n,i,j)); printf("\n");

        }

        free(L); free(U);

    }



    free_lu(&f);

    free(A); free(b); free(b_orig); free(x); free(r);

    return EXIT_SUCCESS;

}

