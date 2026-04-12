# Equation-solver-INFINITE-
AN UPGRADE, after learning some shitty linear algebra i thought about getting what i studied into a code, README file is important try reading it
# Linear System Solver (LU + QR + Condition Number Estimator)

## EXPLINATION

This project is a numerical linear algebra solver written in C.

It solves systems of linear equations:

A x = b

where:
- A is an n × n matrix
- x is the unknown vector
- b is the right-hand side vector

Unlike basic solvers, this implementation includes:

- LU decomposition with scaled partial pivoting
- Householder QR factorization with column pivoting
- System classification (unique / infinite / no solution)
- Iterative refinement for improved accuracy
- Condition number estimation (1-norm, LAPACK-style approach)
- Numerical rank detection using QR factorization

---

## Key Features

### 1. LU Decomposition (Doolittle Method)

The system is factorized as:

A = P * L * U

Where:
- P = permutation matrix (row swaps)
- L = lower triangular matrix (unit diagonal)
- U = upper triangular matrix

Scaled pivoting is used to improve numerical stability by avoiding small pivot errors.

---

### 2. Linear System Solver

After factorization, the system is solved using:

Forward substitution:
L y = P b

Back substitution:
U x = y

This produces the solution vector x.

---

### 3. Iterative Refinement

To improve numerical accuracy:

1. Compute residual:
r = b - A x

2. Solve correction system:
A dx = r

3. Update solution:
x = x + dx

This reduces floating-point error, especially for ill-conditioned systems.

---

### 4. QR Factorization (Rank Detection)

The implementation uses Householder QR decomposition with column pivoting:

A = Q R

Used for numerical rank detection.

- Column pivoting improves stability
- Rank is determined by small diagonal values of R

---

### 5. System Classification

The solver classifies systems using rank:

Unique solution:
rank(A) = n

Infinite solutions:
rank(A) = rank([A|b]) < n

No solution:
rank(A) < rank([A|b])

---

### 6. Condition Number Estimation

The condition number is estimated as:

kappa_1(A) = ||A||_1 * ||A^-1||_1

Where:
- ||A||_1 = maximum column sum norm
- ||A^-1||_1 = estimated using a Hager-style iterative method

This measures sensitivity to numerical errors.

---

### 7. Determinant Calculation

The determinant is computed from LU:

det(A) = sign(P) * product(diagonal(U))

---

## Numerical Stability Techniques

This solver is designed for robustness:

- Machine epsilon awareness (EPS)
- Scaled partial pivoting
- Householder reflections for QR stability
- Rank thresholds based on floating-point precision
- Iterative refinement loop
- Norm-based stopping conditions

---

## Performance

- LU decomposition: O(n^3)
- QR factorization: O(n^3)
- Condition estimation: O(n^2 to n^3)
- Memory usage: O(n^2)

Supports systems up to:

n <= 10000 (theoretical, RAM dependent)

---

## Input Format

The program expects an augmented matrix:

a11 a12 a13 ... a1n b1
a21 a22 a23 ... a2n b2
...
an1 an2 an3 ... ann bn

---

## Output

- Solution vector x
- Rank of matrix
- Determinant
- Condition number estimate
- Residual error ||Ax - b||
- (Optional) LU matrices for small systems

---

## Why this project is interesting

This is not just a solver — it combines:

- Classical numerical linear algebra (Golub & Van Loan methods)
- Practical stability techniques used in LAPACK
- Real-world floating-point robustness strategies

It essentially recreates a mini scientific computing backend in pure C.






P.S This code and this overview was made by claude, i ordered the math and it gave it to me on a golden plate, EqSol(inf)-O4 is by chatgpt and its not the better version, the other version on the other hand is by claude which was almost perfect but it has some flaws (MATHMATECALLY) and they don't make any problem but some of them are (MATHMATECALLY) wrong
