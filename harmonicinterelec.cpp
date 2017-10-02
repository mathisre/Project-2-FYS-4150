#include <armadillo>
#include <cstdlib>
#include <math.h>
#include <time.h>
using namespace arma;
using namespace std;

double offdiag(double ** A, int n, int * k, int * l)
{ // Find largest offdiagonal matrix element with indices k,l in A
    double max =0;
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            if (i!=j && fabs(A[i][j])>max){
                max = fabs(A[i][j]);
                *l = i;
                *k = j;
            }}}
    return max;
}

int main()
{
    //Create n, rho and matrix element d and e
    int n = 200;
    clock_t start, finish;
    vec rho = linspace<vec>(pow(10,-5), 30, n);
    double h = (rho(n-1) - rho(0))/n;
    double w = 0.01 ;
    double e = -pow(h,-2); //offdiagonal
    vec d = ones<vec>(n,1)*2*pow(h,-2) + w*w*pow(rho,2) + 1 / rho; //diagonal element
    cout << "w =   " << w << endl;

    //Create matrices A and B
    //A is for the Jacobi algorithm and B is for the armadillo eig_sym function
    double **  A;
    A = new double*[n];
    for (int i = 0; i < n; i++){
        A[i] = new double[n];
    }
    mat B = zeros<mat>(n,n);
    for (int i=0; i<n; i++){
        if(i!=n-1){
            A[i][i] = d(i);
            A[i][i+1] = e;
            B(i,i) = d(i);
            B(i,i+1) = e;
        }
        if (i !=0 ){
            A[i][i]=d(i);
            A[i][i-1] = e;
            B(i,i)=d(i);
            B(i,i-1) = e;
        }
    }

    //Unit test to make sure the offdiag function is working properly
    //Define a matrix C with known maximum offdiagonal element
    //and check that the function gives the same result
    int p,o;
    double  ** C;
    C = new double*[3];
    for (int i = 0; i < 3; i++){
        C[i] = new double[3];
    }
    C[0][0] = 1, C[0][1] = 4, C[0][2] = 10, C[1][0] = 2, C[1][1] = 22, C[1][2] = 7, C[2][0] = 6, C[2][1] = 36, C[2][2] = 91;
    double maxtest = offdiag(C, 3, &p, &o);
    if (maxtest != 36 && p == 2 && o == 1 ){ // The matrix has largest offdiagonal element in C[2][1]
        cout << "Something is wrong with the max offdiagonal function" << endl;
        exit(EXIT_FAILURE);
    }

    //Define parameters needed for Jacobi rotation algorithm
    double tol = pow(10,-7);
    int k, l;
    mat V = eye<mat>(n,n);
    int max_iter = n*n*n;
    int iter = 0;
    double offdiag_max = offdiag(A, n, &k, &l);
    start = clock();

    //Jacobi algorithm
    while (offdiag_max > tol && iter < max_iter){
        offdiag_max = offdiag(A, n, &k, &l);
        double c, s, t, tau;
        tau = (A[l][l] - A[k][k])/(2*A[k][l]);

        if ( tau > 0 ) {
            t = 1.0/(tau + pow(1+pow(tau,2),0.5));
        } else {
            t = -1.0/( -tau + pow(1+pow(tau,2),0.5));
        }

        c = pow(1+pow(t,2),-0.5);
        s = t*c;

        double akk = A[k][k];
        double all = A[l][l];
        A[k][k] = akk*pow(c,2) - 2*A[k][l]*c*s + all*pow(s,2); //matrix transformation on diagonal
        A[l][l] = all*pow(c,2) + 2*A[k][l]*c*s + akk*pow(s,2); //and k,l off diagonal
        A[k][l]=0;
        A[l][k]=0;
        for (int i=0; i<n; i++){
            if (i!=k && i!=l){ //already changed these elements
                double a_ik = A[i][k];
                double a_il = A[i][l];
                A[i][k] = c*a_ik - s*a_il; //matrix transformation on other offdiagonals
                A[k][i] = A[i][k];
                A[i][l] = c*a_il + s*a_ik;
                A[l][i] = A[i][l];
            }
            double vik = V(i,k);
            double vil = V(i,l);
            V(i,k) = c*vik - s*vil;
            V(i,l) = c*vil + s*vik;
        } //Unit test to check orthogonality in eigenvectors
        if (dot(V.col(k),V.col(l)) > 0 + tol && dot(V.col(k),V.col(l)) < 0 - tol){
            cout << "Eigenvectors are not orthogonal!" << endl;
            exit(EXIT_FAILURE);
        } //Unit test to check length of eigenvectors. Should be one.
        if ( norm(V.col(k)) > 0 + tol && norm(V.col(k)) < 0 - tol && norm(V.col(l)) > 0 + tol && norm(V.col(l)) < 0 - tol){
            cout << "Eigenvector does not have unit length" << endl;
            exit(EXIT_FAILURE);
        }
        iter +=1;
    }
    finish = clock();

    cout << "Number of iterations: " << iter << "\n";

    // Find three lowest eigenvalues from A.
    vec minfind = ones<vec>(3)*100000000;
    int a, a1, a2, a3;
    for (int m=0; m<3; m++){
        for (int i=0;i<n;i++){
            if (m==0 && (fabs(A[i][i]) < minfind(m)) &&  fabs(A[i][i]) != minfind(m)){
                minfind(m) = fabs(A[i][i]);
                a = i;
                a1 = i;
            }
            if (m==1 && fabs(A[i][i]) < minfind(m) && fabs(A[i][i]) != minfind(m)){
                minfind(m) = fabs(A[i][i]);
                a = i;
                a2 = i;
            }
            if (m==2 && fabs(A[i][i]) < minfind(m) &&  fabs(A[i][i]) != minfind(m)){
                minfind(m) = fabs(A[i][i]);
                a = i;
                a3 = i;
            }
        }

        A[a][a] = 100000000; //To make sure the same eigenvalue is not counted twice
    }

    for (int i=0;i<3;i++){
        cout << "Eig_Jac("<< i<< ") = " << minfind(i) << "      ";
    }

    //Armadillo eig_sym solution of eigenvalue and eigenvectors
    vec eigval; mat eigvec;
    eig_sym(eigval,eigvec,B);
    cout << endl;
    for (int i=0;i<3;i++){
        cout << "Eigval("<< i<< ")  = " << eigval(i) << "      ";
    }

    double ftime = double (finish-start)/CLOCKS_PER_SEC;
    cout<< endl << "time = " << ftime<< "s"<< endl;
    return 0;
}

