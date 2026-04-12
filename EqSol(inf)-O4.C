#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <math.h>

#include <float.h>



#define EPS           2.220446049250313e-16

#define REFINE_MAX    10

#define REFINE_TOL    (EPS * 4.0)

#define MAX_N         10000

#define RANK_TOL_FACTOR 2.0



typedef enum {

    SOLVE_OK = 0,

    SOLVE_INFINITE = -1,

    SOLVE_NONE = -2,

    SOLVE_DEGENERATE = -3

} SolveResult;



typedef struct {

    int n;

    double *lu;

    int *piv;

    double *scales;

    int rank;

    int sign;

} LU;



typedef struct {

    int n;

    double *qr;

    int *jpvt;

    double *tau;

    double *work;

    double *cnorm;

    int rank;

} QR;



#define A_(a,n,i,j) ((a)[(i)*(n)+(j)])

#define L_(f,i,j)   ((f).lu[(i)*(f).n+(j)])



/* ================= MEMORY ================= */



static void *safe_malloc(size_t sz){

    void *p = malloc(sz);

    if(!p){fprintf(stderr,"malloc fail\n"); exit(1);}

    return p;

}



static double *vec(int n){ return (double*)safe_malloc(sizeof(double)*n); }

static double *mat(int n){ return (double*)safe_malloc(sizeof(double)*n*n); }



static LU alloc_lu(int n){

    LU f;

    f.n=n;

    f.lu=mat(n);

    f.piv=(int*)safe_malloc(sizeof(int)*n);

    f.scales=vec(n);

    f.rank=0;

    f.sign=1;

    return f;

}



static QR alloc_qr(int n){

    QR f;

    f.n=n;

    f.qr=mat(n);

    f.jpvt=(int*)safe_malloc(sizeof(int)*n);

    f.tau=vec(n);

    f.work=vec(n);

    f.cnorm=vec(n);

    f.rank=0;

    return f;

}



/* ================= UTIL ================= */



static double inf_norm(double *v,int n){

    double m=0;

    for(int i=0;i<n;i++){

        double a=fabs(v[i]);

        if(a>m)m=a;

    }

    return m;

}



static double mat_1norm(const double *A,int n){

    double mx=0;

    for(int j=0;j<n;j++){

        double s=0;

        for(int i=0;i<n;i++)

            s+=fabs(A_(A,n,i,j));

        if(s>mx) mx=s;

    }

    return mx;

}



/* ================= HOUSEHOLDER ================= */



static void householder(double *x,int n,double *tau){

    if(n<=1){*tau=0; return;}



    double sigma=0;

    for(int i=1;i<n;i++) sigma+=x[i]*x[i];



    if(sigma<EPS*EPS){*tau=0; return;}



    double norm=sqrt(x[0]*x[0]+sigma);

    double beta=(x[0]>0)?-norm:norm;



    double inv=1.0/(x[0]-beta);

    for(int i=1;i<n;i++) x[i]*=inv;



    *tau=(beta-x[0])/beta;

    x[0]=beta;

}



/* ================= LU ================= */



static int lu_factor(LU *f){

    int n=f->n;

    int rank=0;

    int sign=1;



    double maxA=0;

    for(int i=0;i<n*n;i++)

        if(fabs(f->lu[i])>maxA) maxA=fabs(f->lu[i]);



    double tol=RANK_TOL_FACTOR*n*EPS*maxA;



    for(int i=0;i<n;i++){

        double mx=0;

        for(int j=0;j<n;j++){

            double v=fabs(L_(f,i,j));

            if(v>mx) mx=v;

        }

        f->scales[i]=(mx==0)?1:mx;

        f->piv[i]=i;

    }



    for(int k=0;k<n;k++){



        int piv=k;

        double best=fabs(L_(f,k,k))/f->scales[k];



        for(int i=k+1;i<n;i++){

            double v=fabs(L_(f,i,k))/f->scales[i];

            if(v>best){best=v;piv=i;}

        }



        double pivot=fabs(L_(f,piv,k))/f->scales[piv];

        if(pivot<tol) continue;



        rank++;



        if(piv!=k){

            for(int j=0;j<n;j++){

                double t=L_(f,k,j);

                L_(f,k,j)=L_(f,piv,j);

                L_(f,piv,j)=t;

            }

            int ti=f->piv[k];

            f->piv[k]=f->piv[piv];

            f->piv[piv]=ti;

            sign=-sign;

        }



        double diag=L_(f,k,k);



        for(int i=k+1;i<n;i++){

            double fac=L_(f,i,k)/diag;

            L_(f,i,k)=fac;

            for(int j=k+1;j<n;j++)

                L_(f,i,j)-=fac*L_(f,k,j);

        }

    }



    f->rank=rank;

    f->sign=sign;

    return rank;

}



/* ================= LU SOLVE ================= */



static SolveResult lu_solve(const LU *f,const double *b,double *x){

    int n=f->n;

    double *y=vec(n);



    for(int i=0;i<n;i++)

        y[i]=b[f->piv[i]];



    for(int i=0;i<n;i++)

        for(int j=0;j<i;j++)

            y[i]-=L_(f,i,j)*y[j];



    for(int i=n-1;i>=0;i--){

        double d=L_(f,i,i);

        if(fabs(d)<EPS){free(y); return SOLVE_DEGENERATE;}

        for(int j=i+1;j<n;j++)

            y[i]-=L_(f,i,j)*x[j];

        x[i]=y[i]/d;

    }



    free(y);

    return SOLVE_OK;

}



/* ================= TRANSPOSE SOLVE (FIXED) ================= */



static void lu_solve_T(const LU *f,const double *b,double *x){

    int n=f->n;

    double *y=vec(n);



    for(int i=0;i<n;i++)

        y[i]=b[i];



    for(int i=0;i<n;i++){

        for(int j=0;j<i;j++)

            y[i]-=L_(f,j,i)*y[j];

        y[i]/=L_(f,i,i);

    }



    for(int i=n-1;i>=0;i--){

        for(int j=i+1;j<n;j++)

            y[i]-=L_(f,i,j)*y[j];

    }



    for(int i=0;i<n;i++)

        x[f->piv[i]]=y[i];



    free(y);

}



/* ================= REFINEMENT ================= */



static void refine(const double *A,const LU *f,const double *b,double *x){

    int n=f->n;

    double *r=vec(n);

    double *dx=vec(n);



    for(int it=0;it<REFINE_MAX;it++){

        for(int i=0;i<n;i++){

            double s=b[i];

            for(int j=0;j<n;j++)

                s-=A_(A,n,i,j)*x[j];

            r[i]=s;

        }



        if(inf_norm(r,n)<REFINE_TOL) break;



        lu_solve(f,r,dx);



        for(int i=0;i<n;i++)

            x[i]+=dx[i];

    }



    free(r); free(dx);

}



/* ================= CONDITION (FIXED SIMPLE) ================= */



static double inv_1norm(const LU *f){

    int n=f->n;

    double *v=vec(n),*w=vec(n),*s=vec(n),*z=vec(n);



    for(int i=0;i<n;i++) v[i]=1.0/n;



    double est=0;



    for(int it=0;it<6;it++){

        lu_solve(f,v,w);



        double norm=0;

        for(int i=0;i<n;i++) norm+=fabs(w[i]);



        if(norm>est) est=norm;



        for(int i=0;i<n;i++)

            s[i]=(w[i]>=0)?1:-1;



        lu_solve_T(f,s,z);



        int jmax=0;

        double m=fabs(z[0]);

        for(int i=1;i<n;i++)

            if(fabs(z[i])>m){m=fabs(z[i]); jmax=i;}



        memset(v,0,sizeof(double)*n);

        v[jmax]=1;

    }



    free(v);free(w);free(s);free(z);

    return est;

}



static double cond(const double *A,const LU *f){

    return mat_1norm(A,f->n)*inv_1norm(f);

}



/* ================= QR (FIXED STABLE) ================= */



static void qr(QR *f){

    int n=f->n;



    for(int j=0;j<n;j++){

        f->jpvt[j]=j;

        f->cnorm[j]=0;

        for(int i=0;i<n;i++)

            f->cnorm[j]+=QR_(f,i,j)*QR_(f,i,j);

    }



    int rank=0;

    double maxd=0;



    for(int k=0;k<n;k++){



        int p=k;

        double best=f->cnorm[k];



        for(int j=k+1;j<n;j++)

            if(f->cnorm[j]>best){best=f->cnorm[j];p=j;}



        if(best<EPS*EPS) break;



        if(p!=k){

            for(int i=0;i<n;i++){

                double t=QR_(f,i,k);

                QR_(f,i,k)=QR_(f,i,p);

                QR_(f,i,p)=t;

            }

        }

        for(int i=k;i<n;i++)

            f->work[i-k]=QR_(f,i,k);



        householder(f->work,n-k,&f->tau[k]);



        if(k==0) maxd=fabs(f->work[0]);



        if(fabs(f->work[0])<RANK_TOL_FACTOR*n*EPS*maxd)

            break;



        rank++;



        QR_(f,k,k)=f->work[0];

        for(int i=k+1;i<n;i++)

            QR_(f,i,k)=f->work[i-k];



        for(int j=k+1;j<n;j++){

            double dot=QR_(f,k,j);

            for(int i=k+1;i<n;i++)

                dot+=QR_(f,i,k)*QR_(f,i,j);



            dot*=f->tau[k];



            QR_(f,k,j)-=dot;

            for(int i=k+1;i<n;i++)

                QR_(f,i,j)-=dot*QR_(f,i,k);



            f->cnorm[j]=0;

            for(int i=k+1;i<n;i++)

                f->cnorm[j]+=QR_(f,i,j)*QR_(f,i,j);

        }

    }



    f->rank=rank;

}



/* ================= MAIN ================= */



int main(){

    int n;

    scanf("%d",&n);



    double *A=mat(n),*b=vec(n),*x=vec(n);



    for(int i=0;i<n;i++)

        for(int j=0;j<n;j++)

            scanf("%lf",&A_(A,n,i,j));



    for(int i=0;i<n;i++)

        scanf("%lf",&b[i]);



    LU f=alloc_lu(n);

    memcpy(f.lu,A,sizeof(double)*n*n);



    lu_factor(&f);



    lu_solve(&f,b,x);



    refine(A,&f,b,x);



    printf("solution:\n");

    for(int i=0;i<n;i++)

        printf("%lf\n",x[i]);



    printf("cond ~ %e\n",cond(A,&f));



    free(f.lu);free(f.piv);free(f.scales);

    free(A);free(b);free(x);

}
