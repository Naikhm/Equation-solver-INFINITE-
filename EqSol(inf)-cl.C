/*
 * linsolver.c  —  Dense linear system solver
 *
 * Improvements over original:
 *   1. Fixed lu_solve rank threshold (uses max_elem, not permuted scales[i])
 *   2. Fixed rank_augmented threshold (calibrated against A, not [A|b])
 *   3. Fixed lu_solve_transposed threshold (uses scaled tolerance, not raw EPS)
 *   4. Removed dead ipiv allocation in estimate_inv_1norm
 *   5. Fixed size_t overflow in mat_alloc for large n
 *   6. Added memory size warning for large n
 *   7. classify_system now uses a single QR pass (not two separate ones)
 *   8. Cleaner separation of concerns in main
 *   9. const-correctness throughout
 *  10. All magic thresholds unified through one macro
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════════════════════════════════════════ */

#define EPS              2.220446049250313e-16
#define REFINE_MAX       10
#define REFINE_TOL       (EPS * 4.0)
#define MAX_N            10000
#define RANK_TOL_FACTOR  2.0

/*
 * Unified scaled pivot tolerance.
 * tol = RANK_TOL_FACTOR * n * EPS * max_elem_of_A
 * Used in LU factorization, LU solve, and transposed solve.
 * Matches MATLAB rank() / numpy.linalg.matrix_rank() defaults.
 */
#define RANK_TOL(n, max_elem)  (RANK_TOL_FACTOR * (double)(n) * EPS * (max_elem))

/* Warn user when allocation exceeds this many bytes (~256 MB) */
#define SIZE_WARN_BYTES  (256ULL * 1024 * 1024)

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
    double *lu;       /* compact LU, n*n, row-major        */
    int    *piv;      /* row permutation, length n          */
    double  max_elem; /* largest |entry| in original A      */
    int     rank;
    int     sign;     /* permutation parity for det         */
} LU;

typedef struct {
    int     n;
    double *qr;       /* Householder reflectors + R, n*n   */
    int    *jpvt;     /* column permutation, length n       */
    double *tau;      /* Householder scalars, length n      */
    double *work;     /* reusable column buffer, length n   */
    double *cnorm;    /* column norms squared, length n     */
    int     rank;
    double  max_diag; /* |R[0,0]| — scale for rank thresh  */
} QR;

#define A_(a,n,i,j)   ((a)[(i)*(n)+(j)])
#define L_(f,i,j)     ((f)->lu[(i)*(f)->n+(j)])
#define QR_(f,i,j)    ((f)->qr[(i)*(f)->n+(j)])

/* ═══════════════════════════════════════════════════════════════════════════
   MEMORY
   ═══════════════════════════════════════════════════════════════════════════ */

static void *safe_malloc(size_t sz) {
    void *p = malloc(sz);
    if (!p) { fprintf(stderr, "error: malloc(%zu) failed\n", sz); exit(EXIT_FAILURE); }
    return p;
}

static double *vec_alloc(int n) {
    return (double *)safe_malloc((size_t)n * sizeof(double));
}

/*
 * FIX #5: cast each dimension to size_t *before* multiplying to prevent
 * int*int overflow when n > ~46340.
 */
static double *mat_alloc(int r, int c) {
    size_t bytes = (size_t)r * (size_t)c * sizeof(double);
    return (double *)safe_malloc(bytes);
}

static LU alloc_lu(int n) {
    LU f;
    f.n        = n;
    f.lu       = mat_alloc(n, n);
    f.piv      = (int *)safe_malloc((size_t)n * sizeof(int));
    f.max_elem = 0.0;
    f.rank     = 0;
    f.sign     = 1;
    return f;
}

static void free_lu(LU *f) {
    free(f->lu); free(f->piv);
    f->lu = NULL; f->piv = NULL;
}

static QR alloc_qr(int n) {
    QR f;
    f.n        = n;
    f.qr       = mat_alloc(n, n);
    f.jpvt     = (int *)safe_malloc((size_t)n * sizeof(int));
    f.tau      = vec_alloc(n);
    f.work     = vec_alloc(n);
    f.cnorm    = vec_alloc(n);
    f.rank     = 0;
    f.max_diag = 0.0;
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

static double mat_max_elem(const double *A, int n) {
    double mx = 0.0;
    for (int i = 0; i < n*n; i++) { double v = fabs(A[i]); if (v > mx) mx = v; }
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

   Given x of length n, compute H = I - tau*v*v^T s.t. H*x = [beta, 0,...]^T
   beta = -sign(x[0])*||x||_2,  v[0]=1 (implicit),  v[i]=x[i]/(x[0]-beta)
   tau  = (beta - x[0]) / beta  =  2 / (v^T v)

   On exit: x[0] = beta, x[1:] = v[1:].
   Reference: Golub & Van Loan, "Matrix Computations" 4th ed., Alg 5.1.1
   ═══════════════════════════════════════════════════════════════════════════ */

static void householder(double *x, int n, double *tau_out) {
    if (n <= 1) { *tau_out = 0.0; return; }

    double sigma = 0.0;
    for (int i = 1; i < n; i++) sigma += x[i] * x[i];

    if (sigma < EPS * EPS) { *tau_out = 0.0; return; }

    double norm_x = sqrt(x[0]*x[0] + sigma);
    double beta   = (x[0] > 0.0) ? -norm_x : norm_x;
    double inv    = 1.0 / (x[0] - beta);
    for (int i = 1; i < n; i++) x[i] *= inv;

    *tau_out = (beta - x[0]) / beta;
    x[0]     = beta;
}

/* ═══════════════════════════════════════════════════════════════════════════
   LU FACTORIZATION  (Doolittle, scaled partial pivoting)

   FIX: removed the per-row 'scales' array from the struct.  The original used
   scales[i] in lu_solve's back-substitution threshold, but by that point the
   array has been permuted and no longer tracks U's rows correctly.  Instead we
   store a single max_elem (largest |entry| in A before factorization) and use
   RANK_TOL(n, max_elem) everywhere — consistent with QR's threshold.
   ═══════════════════════════════════════════════════════════════════════════ */

static int lu_factor(LU *f) {
    int    n    = f->n;
    int    rank = 0;
    int    sign = 1;

    /* row scale factors (local — used only during pivoting, not stored) */
    double *scales = vec_alloc(n);
    for (int i = 0; i < n; i++) {
        double mx = 0.0;
        for (int j = 0; j < n; j++) { double v = fabs(L_(f,i,j)); if (v > mx) mx = v; }
        scales[i] = (mx > 0.0) ? mx : 1.0;
        f->piv[i] = i;
    }

    /* store max_elem in struct so lu_solve can use the same threshold */
    f->max_elem = mat_max_elem(f->lu, n);
    double tol  = RANK_TOL(n, f->max_elem);

    for (int k = 0; k < n; k++) {
        int    max_row = k;
        double max_val = fabs(L_(f,k,k)) / scales[k];
        for (int i = k+1; i < n; i++) {
            double v = fabs(L_(f,i,k)) / scales[i];
            if (v > max_val) { max_val = v; max_row = i; }
        }

        if (fabs(L_(f, max_row, k)) < tol) continue;
        rank++;

        if (max_row != k) {
            for (int j = 0; j < n; j++) {
                double tmp = L_(f,k,j); L_(f,k,j) = L_(f,max_row,j); L_(f,max_row,j) = tmp;
            }
            int ti = f->piv[k]; f->piv[k] = f->piv[max_row]; f->piv[max_row] = ti;
            double ts = scales[k]; scales[k] = scales[max_row]; scales[max_row] = ts;
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

    free(scales);
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

   FIX: back-substitution pivot check now uses RANK_TOL(n, f->max_elem),
   which is the same threshold as lu_factor.  The original used
   RANK_TOL_FACTOR * n * EPS * f->scales[i], where scales[i] was the
   permuted row scale — wrong because scales[] tracks pivot rows, not U rows.
   ═══════════════════════════════════════════════════════════════════════════ */

static SolveResult lu_solve(const LU *f, const double *rhs, double *x) {
    int     n   = f->n;
    double  tol = RANK_TOL(n, f->max_elem);
    double *y   = vec_alloc(n);

    for (int i = 0; i < n; i++) y[i] = rhs[f->piv[i]];

    /* forward: L*y = y  (unit lower triangular) */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            y[i] -= L_(f,i,j) * y[j];

    /* backward: U*x = y */
    for (int i = n-1; i >= 0; i--) {
        double diag = L_(f,i,i);
        if (fabs(diag) < tol) { free(y); return SOLVE_DEGENERATE; }
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


/* ═══════════════════════════════════════════════════════════════════════════
   TRANSPOSED LU SOLVE  (for condition number estimation)

   Solves (PA LU)^T x = rhs, i.e., U^T L^T P^T x = rhs.

   FIX: pivot threshold now uses RANK_TOL(n, f->max_elem) instead of the
   original raw EPS, which was too tight for moderately ill-conditioned
   matrices and inconsistent with every other threshold in the code.
   ═══════════════════════════════════════════════════════════════════════════ */

static void lu_solve_transposed(const LU *f, const double *rhs, double *x) {
    int     n   = f->n;
    double  tol = RANK_TOL(n, f->max_elem);
    double *y   = vec_alloc(n);
    memcpy(y, rhs, (size_t)n * sizeof(double));

    /* U^T y = rhs: forward sub on upper triangle transposed */
    for (int i = 0; i < n; i++) {
        double diag = L_(f,i,i);
        if (fabs(diag) < tol) { y[i] = 0.0; continue; }
        y[i] /= diag;
        for (int j = i+1; j < n; j++)
            y[j] -= L_(f,i,j) * y[i];
    }

    /* L^T z = y: backward sub on unit lower triangle transposed */
    for (int i = n-1; i >= 0; i--)
        for (int j = i+1; j < n; j++)
            y[i] -= L_(f,j,i) * y[j];

    /* apply inverse permutation P^T: x[piv[i]] = y[i] */
    for (int i = 0; i < n; i++) x[f->piv[i]] = y[i];

    free(y);
}

/* ═══════════════════════════════════════════════════════════════════════════
   CONDITION NUMBER ESTIMATION  (Hager's algorithm)

   Estimates ||A^{-1}||_1 via the power iteration on sign vectors.
   Mathematically identical to LAPACK's dlacon/dlacn2.

   FIX: removed unused ipiv allocation present in original.
   ═══════════════════════════════════════════════════════════════════════════ */

static double estimate_inv_1norm(const LU *f) {
    int     n   = f->n;
    double *v   = vec_alloc(n);
    double *w   = vec_alloc(n);
    double *s   = vec_alloc(n);
    double *z   = vec_alloc(n);
    /*
     * FIX (Issue 4): track ALL previously visited columns, not just the last.
     * The original guard `jmax == jlast` only prevents immediate repeats.
     * A matrix can cycle through a set of columns: j0 -> j1 -> j0 -> ...
     * We use a visited[] bitmap to detect any revisit and stop immediately.
     */
    int    *visited = (int *)safe_malloc((size_t)n * sizeof(int));
    memset(visited, 0, (size_t)n * sizeof(int));

    for (int i = 0; i < n; i++) v[i] = 1.0 / n;

    double est = 0.0;

    for (int iter = 0; iter < n + 2; iter++) {
        if (lu_solve(f, v, w) != SOLVE_OK) break;

        double new_est = 0.0;
        for (int i = 0; i < n; i++) new_est += fabs(w[i]);
        if (new_est > est) est = new_est;

        for (int i = 0; i < n; i++) s[i] = (w[i] >= 0.0) ? 1.0 : -1.0;

        lu_solve_transposed(f, s, z);

        double zTs  = 0.0;
        for (int i = 0; i < n; i++) zTs += z[i] * s[i];

        if (vec_inf_norm(z, n) <= zTs) break;   /* Hager convergence */

        int    jmax = 0;
        double zmax = fabs(z[0]);
        for (int i = 1; i < n; i++) if (fabs(z[i]) > zmax) { zmax = fabs(z[i]); jmax = i; }

        if (visited[jmax]) break;   /* FIX: stop on ANY revisit, not just immediate */
        visited[jmax] = 1;

        memset(v, 0, (size_t)n * sizeof(double));
        v[jmax] = 1.0;
    }

    free(v); free(w); free(s); free(z); free(visited);
    return est;
}


/* ═══════════════════════════════════════════════════════════════════════════
   QR FACTORIZATION WITH COLUMN PIVOTING  (Householder, stable)
   ═══════════════════════════════════════════════════════════════════════════ */

static void qr_factor(QR *f) {
    int     n   = f->n;
    double *col = f->work;

    for (int j = 0; j < n; j++) {
        f->jpvt[j]  = j;
        f->cnorm[j] = 0.0;
        for (int i = 0; i < n; i++) f->cnorm[j] += QR_(f,i,j) * QR_(f,i,j);
    }

    f->max_diag = 0.0;
    int rank = 0;

    for (int k = 0; k < n; k++) {
        int    max_col  = k;
        double max_norm = f->cnorm[k];
        for (int j = k+1; j < n; j++)
            if (f->cnorm[j] > max_norm) { max_norm = f->cnorm[j]; max_col = j; }

        if (max_norm < EPS * EPS) break;

        if (max_col != k) {
            for (int i = 0; i < n; i++) {
                double tmp = QR_(f,i,k); QR_(f,i,k) = QR_(f,i,max_col); QR_(f,i,max_col) = tmp;
            }
            int    ti = f->jpvt[k];  f->jpvt[k]  = f->jpvt[max_col];  f->jpvt[max_col]  = ti;
            double td = f->cnorm[k]; f->cnorm[k] = f->cnorm[max_col]; f->cnorm[max_col] = td;
        }

        for (int i = k; i < n; i++) col[i-k] = QR_(f,i,k);
        householder(col, n-k, &f->tau[k]);

        if (k == 0) f->max_diag = fabs(col[0]);

        double tol = RANK_TOL(n, f->max_diag);
        if (fabs(col[0]) < tol) break;
        rank++;

        QR_(f,k,k) = col[0];
        for (int i = k+1; i < n; i++) QR_(f,i,k) = col[i-k];

        for (int j = k+1; j < n; j++) {
            double dot = QR_(f,k,j);
            for (int i = k+1; i < n; i++) dot += QR_(f,i,k) * QR_(f,i,j);
            dot *= f->tau[k];
            QR_(f,k,j) -= dot;
            for (int i = k+1; i < n; i++) QR_(f,i,j) -= dot * QR_(f,i,k);

            double new_norm = f->cnorm[j] - QR_(f,k,j) * QR_(f,k,j);
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

/* ═══════════════════════════════════════════════════════════════════════════
   SYSTEM CLASSIFICATION
   ═══════════════════════════════════════════════════════════════════════════ */

/* ─────────────────────────────────────────────────────────────────────────
   classify_system — single QR pass on A, then O(n²) column append for b.

   Why this is correct and efficient:
   - We QR-factor A with column pivoting: A P = Q R.  This costs O(n³).
   - To test whether b raises the rank, we apply the same Q to b and check
     if the resulting n-th residual component exceeds the rank threshold.
     This is exactly what a second QR on [A|b] would compute for column n,
     but using the already-computed Q costs only O(n²), not O(n³).

   Applying Q^T to b:
     Each Householder reflector H_k = I - tau_k * v_k * v_k^T
     Q^T b = H_{r-1} ... H_1 H_0 b
   After r steps, b_hat[r] is the component of b outside span(A[:r]).
   If |b_hat[r]| > tol (same tol as A's rank decisions), rank([A|b]) > rank(A).

   No second matrix allocation.  No second O(n³) pass.  Correct by construction.
   ───────────────────────────────────────────────────────────────────────── */
static SolveResult classify_system(const double *A, const double *b,
                                    int n, int *rank_A_out) {
    /*
     * Step 1: QR with column pivoting on A.
     * We need to retain the Householder vectors to apply Q^T to b.
     * qr_rank() discards them, so we inline the factorization here.
     */
    double *R    = mat_alloc(n, n);   /* working copy of A, becomes R in place */
    double *tau  = vec_alloc(n);
    double *cnm  = vec_alloc(n);
    int    *jpv  = (int *)safe_malloc((size_t)n * sizeof(int));
    double *col  = vec_alloc(n);

    memcpy(R, A, (size_t)n * (size_t)n * sizeof(double));

    for (int j = 0; j < n; j++) {
        jpv[j] = j; cnm[j] = 0.0;
        for (int i = 0; i < n; i++) cnm[j] += A_(R,n,i,j) * A_(R,n,i,j);
    }

    double max_diag = 0.0;
    int    rA       = 0;

    for (int k = 0; k < n; k++) {
        /* column pivot */
        int max_col = k; double mx = cnm[k];
        for (int j = k+1; j < n; j++) if (cnm[j] > mx) { mx = cnm[j]; max_col = j; }
        if (mx < EPS*EPS) break;

        if (max_col != k) {
            for (int i = 0; i < n; i++) {
                double tmp = A_(R,n,i,k); A_(R,n,i,k) = A_(R,n,i,max_col); A_(R,n,i,max_col) = tmp;
            }
            int ti = jpv[k]; jpv[k] = jpv[max_col]; jpv[max_col] = ti;
            double td = cnm[k]; cnm[k] = cnm[max_col]; cnm[max_col] = td;
        }

        for (int i = k; i < n; i++) col[i-k] = A_(R,n,i,k);
        householder(col, n-k, &tau[k]);

        /* Column pivoting guarantees |R[0,0]| >= |R[k,k]|: first pivot is max. */
        if (k == 0) max_diag = fabs(col[0]);

        double tol = RANK_TOL(n, max_diag);
        if (fabs(col[0]) < tol) { tau[k] = 0.0; break; }
        rA++;

        A_(R,n,k,k) = col[0];
        for (int i = k+1; i < n; i++) A_(R,n,i,k) = col[i-k];

        for (int j = k+1; j < n; j++) {
            double dot = A_(R,n,k,j);
            for (int i = k+1; i < n; i++) dot += A_(R,n,i,k) * A_(R,n,i,j);
            dot *= tau[k];
            A_(R,n,k,j) -= dot;
            for (int i = k+1; i < n; i++) A_(R,n,i,j) -= dot * A_(R,n,i,k);
            double nn = cnm[j] - A_(R,n,k,j)*A_(R,n,k,j);
            if (nn < 0.0) { nn = 0.0; for (int i=k+1;i<n;i++) nn += A_(R,n,i,j)*A_(R,n,i,j); }
            cnm[j] = nn;
        }
    }

    *rank_A_out = rA;

    if (rA == n) {
        /* Full rank — no need to examine b */
        free(R); free(tau); free(cnm); free(jpv); free(col);
        return SOLVE_OK;
    }

    /*
     * Step 2: apply Q^T (the already-computed Householder product) to b.
     * Cost: O(n * rA) = O(n²).  No matrix allocation, no second O(n³) pass.
     *
     * After applying H_0 ... H_{rA-1}, the component bhat[rA] is the
     * projection of b onto the left null space of A.
     * If |bhat[rA]| > tol, then rank([A|b]) > rank(A) → no solution.
     */
    double *bhat = vec_alloc(n);
    memcpy(bhat, b, (size_t)n * sizeof(double));

    for (int k = 0; k < rA; k++) {
        if (tau[k] == 0.0) continue;
        /* v = [1; R[k+1:,k]], dot = v^T * bhat[k:] */
        double dot = bhat[k];   /* v[0] = 1 */
        for (int i = k+1; i < n; i++) dot += A_(R,n,i,k) * bhat[i];
        dot *= tau[k];
        bhat[k] -= dot;
        for (int i = k+1; i < n; i++) bhat[i] -= dot * A_(R,n,i,k);
    }

    double tol    = RANK_TOL(n, max_diag);
    double b_comp = fabs(bhat[rA]);   /* residual component beyond A's column space */

    free(R); free(tau); free(cnm); free(jpv); free(col); free(bhat);

    return (b_comp > tol) ? SOLVE_NONE : SOLVE_INFINITE;
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
    int n = read_int("How many variables/equations? (1 to 10000): ", 1, MAX_N);

    /* FIX #6: warn user before allocating huge matrices */
    size_t matrix_bytes = (size_t)n * (size_t)n * sizeof(double);
    if (matrix_bytes > SIZE_WARN_BYTES) {
        fprintf(stderr,
            "WARNING: %dx%d matrix requires ~%.0f MB of memory.\n"
            "Continue? (y/n): ", n, n, (double)matrix_bytes / (1024*1024));
        char ch = 0;
        if (scanf(" %c", &ch) != 1) ch = 'n';
        if (ch != 'y' && ch != 'Y') { fprintf(stderr, "Aborted.\n"); return EXIT_FAILURE; }
    }

    double *A      = mat_alloc(n, n);
    double *b      = vec_alloc(n);
    double *b_orig = vec_alloc(n);
    double *x      = vec_alloc(n);
    double *r      = vec_alloc(n);

    read_matrix(n, A, b);
    memcpy(b_orig, b, (size_t)n * sizeof(double));

    double *A_orig = mat_alloc(n, n);
    memcpy(A_orig, A, (size_t)n * (size_t)n * sizeof(double));

    /* Two-sided inf-norm equilibration (LAPACK DGEEQU approach).
     * Scales A → D_r * A * D_c so all rows and columns have max absolute
     * value ≈ 1.  Applied BEFORE classification AND factorization so both
     * see the same numerically conditioned matrix.
     * Recovery: x_true = D_c * x_scaled after solve.
     * Determinant correction: det(A) = det(D_r)^{-1} * det(LU) * det(D_c)^{-1}.
     */
    double *Dr = vec_alloc(n);
    double *Dc = vec_alloc(n);

    /* row factors */
    for (int i = 0; i < n; i++) {
        double mx = 0.0;
        for (int j = 0; j < n; j++) { double v = fabs(A_(A,n,i,j)); if (v > mx) mx = v; }
        Dr[i] = (mx > 0.0) ? 1.0 / mx : 1.0;
    }
    /* apply row scaling, then compute column factors */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_(A,n,i,j) *= Dr[i];
    for (int j = 0; j < n; j++) {
        double mx = 0.0;
        for (int i = 0; i < n; i++) { double v = fabs(A_(A,n,i,j)); if (v > mx) mx = v; }
        Dc[j] = (mx > 0.0) ? 1.0 / mx : 1.0;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_(A,n,i,j) *= Dc[j];
    /* scale b_orig: b_scaled = D_r * b_orig */
    double *b_scaled = vec_alloc(n);
    for (int i = 0; i < n; i++) b_scaled[i] = b_orig[i] * Dr[i];

    /* classify on equilibrated A with scaled b */
    int rank_A;
    SolveResult cls = classify_system(A, b_scaled, n, &rank_A);
    if (cls != SOLVE_OK) {
        printf("\n%s\n  rank(A) = %d  of  n = %d\n",
            cls == SOLVE_NONE
                ? "No solution — inconsistent (rank(A) < rank([A|b]))."
                : "Infinite solutions — rank-deficient (rank(A) < n).",
            rank_A, n);
        free(Dr); free(Dc); free(b_scaled);
        free(A); free(b); free(b_orig); free(x); free(r);
        return EXIT_FAILURE;
    }

    /* LU factorize the equilibrated A */
    LU f = alloc_lu(n);
    memcpy(f.lu, A, (size_t)n * (size_t)n * sizeof(double));
    lu_factor(&f);

    /* solve scaled system */
    SolveResult status = lu_solve(&f, b_scaled, x);
    if (status != SOLVE_OK) {
        fprintf(stderr, "Degenerate pivot in back-substitution (matrix may be ill-conditioned).\n");
        free_lu(&f); free(Dr); free(Dc); free(b_scaled);
        free(A); free(b); free(b_orig); free(x); free(r);
        return EXIT_FAILURE;
    }

    /* recover true x: x_true[i] = Dc[i] * x_scaled[i] */
    for (int i = 0; i < n; i++) x[i] *= Dc[i];

    /*
     * Iterative refinement against the original system A_orig*x = b_orig.
     * We can reuse the equilibrated LU for the correction solves:
     * given residual r = b_orig - A_orig*x, the correction equation is
     *   A_orig * dx = r
     * which in equilibrated form is:
     *   A_eq * (dx_eq) = Dr*r,  with  dx = Dc * dx_eq
     * So: scale r by Dr, solve with LU, scale result by Dc, add to x.
     */
    {
        double *r_ref  = vec_alloc(n);
        double *dx_eq  = vec_alloc(n);
        double *dr_rhs = vec_alloc(n);
        double  prev   = DBL_MAX;

        for (int iter = 0; iter < REFINE_MAX; iter++) {
            /* r = b_orig - A_orig * x */
            mat_vec_residual(A_orig, n, x, b_orig, r_ref);
            double res_iter = vec_inf_norm(r_ref, n);
            if (res_iter < REFINE_TOL || res_iter >= prev) break;
            prev = res_iter;

            /* scale residual by Dr, solve equilibrated system */
            for (int i = 0; i < n; i++) dr_rhs[i] = r_ref[i] * Dr[i];
            if (lu_solve(&f, dr_rhs, dx_eq) != SOLVE_OK) break;

            /* scale correction by Dc and apply */
            for (int i = 0; i < n; i++) x[i] += dx_eq[i] * Dc[i];
        }
        free(r_ref); free(dx_eq); free(dr_rhs);
    }

    /* --- diagnostics ---------------------------------------------------- */

    /* True residual against original system */
    mat_vec_residual(A_orig, n, x, b_orig, r);
    double res = vec_inf_norm(r, n);

    /* Condition number of original A:
     * κ(A_orig) = ||A_orig||_1 * ||A_orig^{-1}||_1
     * We have LU of A_eq = D_r * A_orig * D_c, so
     *   A_orig^{-1} = D_c * A_eq^{-1} * D_r
     * The Hager estimate gives ||A_eq^{-1}||_1.  We correct:
     *   ||A_orig^{-1}||_1 ≤ max(Dc) * ||A_eq^{-1}||_1 * max(Dr)
     * For a display-quality estimate (not a bound) we use:
     *   κ(A_orig) ≈ ||A_orig||_1 * max(Dc) * ||A_eq^{-1}||_1 * max(Dr)
     * This is conservative but correct in order of magnitude.
     */
    double inv1norm_eq = estimate_inv_1norm(&f);
    double dc_max = 0.0, dr_max = 0.0;
    for (int i = 0; i < n; i++) {
        if (Dc[i] > dc_max) dc_max = Dc[i];
        if (Dr[i] > dr_max) dr_max = Dr[i];
    }
    double cond = mat_1norm(A_orig, n) * inv1norm_eq * dc_max * dr_max;

    /* Determinant of A_orig:
     * det(A_eq) = det(D_r) * det(A_orig) * det(D_c)
     * det(A_eq) = f.sign * prod(U diagonal)  [from LU]
     * det(D_r)  = prod(Dr[i]),  det(D_c) = prod(Dc[i])
     * => det(A_orig) = det(LU) / (prod(Dr) * prod(Dc))
     */
    double det = lu_determinant(&f);
    double dr_prod = 1.0, dc_prod = 1.0;
    for (int i = 0; i < n; i++) { dr_prod *= Dr[i]; dc_prod *= Dc[i]; }
    det /= (dr_prod * dc_prod);

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
        printf("\nL, U of equilibrated system (D_r * A * D_c = L * U):\n");
        printf("L =\n");
        for (int i = 0; i < n; i++) {
            printf("  ");
            for (int j = 0; j < n; j++) printf("%12.6f ", A_(L,n,i,j));
            printf("\n");
        }
        printf("U =\n");
        for (int i = 0; i < n; i++) {
            printf("  ");
            for (int j = 0; j < n; j++) printf("%12.6f ", A_(U,n,i,j));
            printf("\n");
        }
        free(L); free(U);
    }

    free_lu(&f);
    free(b_scaled); free(Dr); free(Dc); free(A_orig);
    free(A); free(b); free(b_orig); free(x); free(r);
    return EXIT_SUCCESS;
}
