#include "Eigen/Eigen"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
using namespace Eigen;
using namespace std;


//PALU
double palu(const Matrix2d& A, const Vector2d& b, const Vector2d& x){
    PartialPivLU<Matrix2d> lu(A);
    Vector2d xx = lu.solve(b);
    double err = (xx-x).norm()/x.norm();   
    return err;
}

//QR
double qr_(const Matrix2d& A, const Vector2d& b, const Vector2d& x){
    HouseholderQR<Matrix2d> qr(A);
    Vector2d xx = qr.solve(b);
    double err = (xx-x).norm()/x.norm();
    return err;
}

int main(){
    Matrix2d A1 {{5.547001962252291e-01,-3.770900990025203e-02}, 
                {8.320502943378437e-01,-9.992887623566787e-01}};
    Vector2d b1 {{-5.169911863249772e-01,1.672384680188350e-01}};

    Matrix2d A2 {{5.547001962252291e-01,-5.540607316466765e-01}, 
                {8.320502943378437e-01,-8.324762492991313e-01}};
    Vector2d b2 {{-6.394645785530173e-04,4.259549612877223e-04}};

    Matrix2d A3 {{5.547001962252291e-01,-5.547001955851905e-01}, 
                {8.320502943378437e-01,-8.320502947645361e-01}};
    Vector2d b3 {{-6.400391328043042e-10,4.266924591433963e-10}};

    Vector2d x {{-1.0e+0,-1.0e+00}}; //vettore soluzione

    cout << setprecision(16) << scientific;
    

    double err1 = palu(A1, b1, x);
    double err2 = palu(A2, b2, x);
    double err3 = palu(A3, b3, x);

    double err1_ = qr_(A1, b1, x);
    double err2_ = qr_(A2, b2, x);
    double err3_ = qr_(A3, b3, x);

    cout << "Errore1, PALU = " << err1 << endl;
    cout << "Errore2, PALU = " << err2<< endl;
    cout << "Errore3, PALU = " << err3<< endl;
    cout << "Errore1, QR = " << err1_<< endl;
    cout << "Errore2, QR = " << err2_<< endl;
    cout << "Errore3, QR = " << err3_<< endl;
    
    





    return 0;
}


