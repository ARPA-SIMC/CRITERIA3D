// downloaded from https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.html
// GNU LGPL license

# include <math.h>
# include <stdbool.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include "eispack.h"

#define DBL_EPSILON 2.2204460492503131e-16 // to not include <float.h>

namespace eigenproblem {


    ////////////////////////////////////////////////////////////////////////////////
    // File: jacobi_cyclic_method.c                                               //
    // Routines:                                                                  //
    //    Jacobi_Cyclic_Method                                                    //
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    //  void Jacobi_Cyclic_Method                                                 //
    //            (double eigenvalues[], double *eigenvectors, double *A, int n)  //
    //                                                                            //
    //  Description:                                                              //
    //     Find the eigenvalues and eigenvectors of a symmetric n x n matrix A    //
    //     using the Jacobi method. Upon return, the input matrix A will have     //
    //     been modified.                                                         //
    //     The Jacobi procedure for finding the eigenvalues and eigenvectors of a //
    //     symmetric matrix A is based on finding a similarity transformation     //
    //     which diagonalizes A.  The similarity transformation is given by a     //
    //     product of a sequence of orthogonal (rotation) matrices each of which  //
    //     annihilates an off-diagonal element and its transpose.  The rotation   //
    //     effects only the rows and columns containing the off-diagonal element  //
    //     and its transpose, i.e. if a[i][j] is an off-diagonal element, then    //
    //     the orthogonal transformation rotates rows a[i][] and a[j][], and      //
    //     equivalently it rotates columns a[][i] and a[][j], so that a[i][j] = 0 //
    //     and a[j][i] = 0.                                                       //
    //     The cyclic Jacobi method considers the off-diagonal elements in the    //
    //     following order: (0,1),(0,2),...,(0,n-1),(1,2),...,(n-2,n-1).  If the  //
    //     the magnitude of the off-diagonal element is greater than a treshold,  //
    //     then a rotation is performed to annihilate that off-diagnonal element. //
    //     The process described above is called a sweep.  After a sweep has been //
    //     completed, the threshold is lowered and another sweep is performed     //
    //     with the new threshold. This process is completed until the final      //
    //     sweep is performed with the final threshold.                           //
    //     The orthogonal transformation which annihilates the matrix element     //
    //     a[k][m], k != m, is Q = q[i][j], where q[i][j] = 0 if i != j, i,j != k //
    //     i,j != m and q[i][j] = 1 if i = j, i,j != k, i,j != m, q[k][k] =       //
    //     q[m][m] = cos(phi), q[k][m] = -sin(phi), and q[m][k] = sin(phi), where //
    //     the angle phi is determined by requiring a[k][m] -> 0.  This condition //
    //     on the angle phi is equivalent to                                      //
    //               cot(2 phi) = 0.5 * (a[k][k] - a[m][m]) / a[k][m]             //
    //     Since tan(2 phi) = 2 tan(phi) / (1.0 - tan(phi)^2),                    //
    //               tan(phi)^2 + 2cot(2 phi) * tan(phi) - 1 = 0.                 //
    //     Solving for tan(phi), choosing the solution with smallest magnitude,   //
    //       tan(phi) = - cot(2 phi) + sgn(cot(2 phi)) sqrt(cot(2phi)^2 + 1).     //
    //     Then cos(phi)^2 = 1 / (1 + tan(phi)^2) and sin(phi)^2 = 1 - cos(phi)^2 //
    //     Finally by taking the sqrts and assigning the sign to the sin the same //
    //     as that of the tan, the orthogonal transformation Q is determined.     //
    //     Let A" be the matrix obtained from the matrix A by applying the        //
    //     similarity transformation Q, since Q is orthogonal, A" = Q'AQ, where Q'//
    //     is the transpose of Q (which is the same as the inverse of Q).  Then   //
    //         a"[i][j] = Q'[i][p] a[p][q] Q[q][j] = Q[p][i] a[p][q] Q[q][j],     //
    //     where repeated indices are summed over.                                //
    //     If i is not equal to either k or m, then Q[i][j] is the Kronecker      //
    //     delta.   So if both i and j are not equal to either k or m,            //
    //                                a"[i][j] = a[i][j].                         //
    //     If i = k, j = k,                                                       //
    //        a"[k][k] =                                                          //
    //           a[k][k]*cos(phi)^2 + a[k][m]*sin(2 phi) + a[m][m]*sin(phi)^2     //
    //     If i = k, j = m,                                                       //
    //        a"[k][m] = a"[m][k] = 0 =                                           //
    //           a[k][m]*cos(2 phi) + 0.5 * (a[m][m] - a[k][k])*sin(2 phi)        //
    //     If i = k, j != k or m,                                                 //
    //        a"[k][j] = a"[j][k] = a[k][j] * cos(phi) + a[m][j] * sin(phi)       //
    //     If i = m, j = k, a"[m][k] = 0                                          //
    //     If i = m, j = m,                                                       //
    //        a"[m][m] =                                                          //
    //           a[m][m]*cos(phi)^2 - a[k][m]*sin(2 phi) + a[k][k]*sin(phi)^2     //
    //     If i= m, j != k or m,                                                  //
    //        a"[m][j] = a"[j][m] = a[m][j] * cos(phi) - a[k][j] * sin(phi)       //
    //                                                                            //
    //     If X is the matrix of normalized eigenvectors stored so that the ith   //
    //     column corresponds to the ith eigenvalue, then AX = X Lamda, where     //
    //     Lambda is the diagonal matrix with the ith eigenvalue stored at        //
    //     Lambda[i][i], i.e. X'AX = Lambda and X is orthogonal, the eigenvectors //
    //     are normalized and orthogonal.  So, X = Q1 Q2 ... Qs, where Qi is      //
    //     the ith orthogonal matrix,  i.e. X can be recursively approximated by  //
    //     the recursion relation X" = X Q, where Q is the orthogonal matrix and  //
    //     the initial estimate for X is the identity matrix.                     //
    //     If j = k, then x"[i][k] = x[i][k] * cos(phi) + x[i][m] * sin(phi),     //
    //     if j = m, then x"[i][m] = x[i][m] * cos(phi) - x[i][k] * sin(phi), and //
    //     if j != k and j != m, then x"[i][j] = x[i][j].                         //
    //                                                                            //
    //  Arguments:                                                                //
    //     double  eigenvalues                                                    //
    //        Array of dimension n, which upon return contains the eigenvalues of //
    //        the matrix A.                                                       //
    //     double* eigenvectors                                                   //
    //        Matrix of eigenvectors, the ith column of which contains an         //
    //        eigenvector corresponding to the ith eigenvalue in the array        //
    //        eigenvalues.                                                        //
    //     double* A                                                              //
    //        Pointer to the first element of the symmetric n x n matrix A. The   //
    //        input matrix A is modified during the process.                      //
    //     int     n                                                              //
    //        The dimension of the array eigenvalues, number of columns and rows  //
    //        of the matrices eigenvectors and A.                                 //
    //                                                                            //
    //  Return Values:                                                            //
    //     Function is of type void.                                              //
    //                                                                            //
    //  Example:                                                                  //
    //     #define N                                                              //
    //     double A[N][N], double eigenvalues[N], double eigenvectors[N][N]       //
    //                                                                            //
    //     (your code to initialize the matrix A )                                //
    //                                                                            //
    //     Jacobi_Cyclic_Method(eigenvalues, (double*)eigenvectors,               //
    //                                                          (double *) A, N); //
    ////////////////////////////////////////////////////////////////////////////////
    //                                                                            //
    //#include <float.h>                           // required for DBL_EPSILON
    //#include <math.h>                            // required for fabs()

    void Jacobi_Cyclic_Method(double* eigenvalues, double *eigenvectors,
                                                                  double *A, int n)
    {

       int i, j, k, m;
       double *pAk, *pAm, *p_r, *p_e;
       double threshold_norm;
       double threshold;
       double tan_phi, sin_phi, cos_phi, tan2_phi, sin2_phi, cos2_phi;
       double sin_2phi, cot_2phi;
       double dum1;
       double dum2;
       double dum3;
       double max;

                      // Take care of trivial cases

       if ( n < 1) return;
       if ( n == 1) {
          eigenvalues[0] = *A;
          *eigenvectors = 1.0;
          return;
       }

              // Initialize the eigenvalues to the identity matrix.

       for (p_e = eigenvectors, i = 0; i < n; i++)
          for (j = 0; j < n; p_e++, j++)
             if (i == j) *p_e = 1.0; else *p_e = 0.0;

                // Calculate the threshold and threshold_norm.

       for (threshold = 0.0, pAk = A, i = 0; i < ( n - 1 ); pAk += n, i++)
          for (j = i + 1; j < n; j++) threshold += *(pAk + j) * *(pAk + j);
       threshold = sqrt(threshold + threshold);
       threshold_norm = threshold * DBL_EPSILON;
       max = threshold + 1.0;
       while (threshold > threshold_norm) {
          threshold /= 10.0;
          if (max < threshold) continue;
          max = 0.0;
          for (pAk = A, k = 0; k < (n-1); pAk += n, k++) {
             for (pAm = pAk + n, m = k + 1; m < n; pAm += n, m++) {
                if ( fabs(*(pAk + m)) < threshold ) continue;

                     // Calculate the sin and cos of the rotation angle which
                     // annihilates A[k][m].

                cot_2phi = 0.5 * ( *(pAk + k) - *(pAm + m) ) / *(pAk + m);
                dum1 = sqrt( cot_2phi * cot_2phi + 1.0);
                if (cot_2phi < 0.0) dum1 = -dum1;
                tan_phi = -cot_2phi + dum1;
                tan2_phi = tan_phi * tan_phi;
                sin2_phi = tan2_phi / (1.0 + tan2_phi);
                cos2_phi = 1.0 - sin2_phi;
                sin_phi = sqrt(sin2_phi);
                if (tan_phi < 0.0) sin_phi = - sin_phi;
                cos_phi = sqrt(cos2_phi);
                sin_2phi = 2.0 * sin_phi * cos_phi;
                //cos_2phi = cos2_phi - sin2_phi;

                         // Rotate columns k and m for both the matrix A
                         //     and the matrix of eigenvectors.

                p_r = A;
                dum1 = *(pAk + k);
                dum2 = *(pAm + m);
                dum3 = *(pAk + m);
                *(pAk + k) = dum1 * cos2_phi + dum2 * sin2_phi + dum3 * sin_2phi;
                *(pAm + m) = dum1 * sin2_phi + dum2 * cos2_phi - dum3 * sin_2phi;
                *(pAk + m) = 0.0;
                *(pAm + k) = 0.0;
                for (i = 0; i < n; p_r += n, i++) {
                   if ( (i == k) || (i == m) ) continue;
                   if ( i < k ) dum1 = *(p_r + k); else dum1 = *(pAk + i);
                   if ( i < m ) dum2 = *(p_r + m); else dum2 = *(pAm + i);
                   dum3 = dum1 * cos_phi + dum2 * sin_phi;
                   if ( i < k ) *(p_r + k) = dum3; else *(pAk + i) = dum3;
                   dum3 = - dum1 * sin_phi + dum2 * cos_phi;
                   if ( i < m ) *(p_r + m) = dum3; else *(pAm + i) = dum3;
                }
                for (p_e = eigenvectors, i = 0; i < n; p_e += n, i++) {
                   dum1 = *(p_e + k);
                   dum2 = *(p_e + m);
                   *(p_e + k) = dum1 * cos_phi + dum2 * sin_phi;
                   *(p_e + m) = - dum1 * sin_phi + dum2 * cos_phi;
                }
             }
             for (i = 0; i < n; i++)
                if ( i == k ) continue;
                else if ( max < fabs(*(pAk + i))) max = fabs(*(pAk + i));
          }
       }
       for (pAk = A, k = 0; k < n; pAk += n, k++) eigenvalues[k] = *(pAk + k);
    }

    /******************************************************************************/

    int bakvec ( int n, double t[], double e[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        BAKVEC determines eigenvectors by reversing the FIGI transformation.

      Discussion:

        BAKVEC forms the eigenvectors of a nonsymmetric tridiagonal
        matrix by back transforming those of the corresponding symmetric
        matrix determined by FIGI.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double T[N*3], contains the nonsymmetric matrix.  Its
        subdiagonal is stored in the positions 2:N of the first column,
        its diagonal in positions 1:N of the second column,
        and its superdiagonal in positions 1:N-1 of the third column.
        T(1,1) and T(N,3) are arbitrary.

        Input/output, double E[N].  On input, E(2:N) contains the
        subdiagonal elements of the symmetric matrix.  E(1) is arbitrary.
        On output, the contents of E have been destroyed.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double Z[N*M], contains the eigenvectors.
        On output, they have been transformed as requested.

        Output, int BAKVEC, an error flag.
        0, for normal return,
        2*N+I, if E(I) is zero with T(I,1) or T(I-1,3) non-zero.
        In this case, the symmetric matrix is not similar
        to the original matrix, and the eigenvectors
        cannot be found by this program.
    */
    {
      int i;
      int ierr;
      int j;

      ierr = 0;

      if ( m == 0 )
      {
        return ierr;
      }

      e[0] = 1.0;
      if ( n == 1 )
      {
        return ierr;
      }

      for ( i = 1; i < n; i++ )
      {
        if ( e[i] == 0.0 )
        {
          if ( t[i+0*3] != 0.0 || t[i-1+2*3] != 0.0 )
          {
            ierr = 2 * n + ( i + 1 );
            return ierr;
          }
          e[i] = 1.0;
        }
        else
        {
          e[i] = e[i-1] * e[i] / t[i-1+2*3];
        }
      }

      for ( j = 0; j < m; j++ )
      {
        for ( i = 1; i < n; i++ )
        {
          z[i+j*n] = z[i+j*n] * e[i];
        }
      }

      return ierr;
    }
    /******************************************************************************/


    void balbak ( int n, int low, int igh, double scale[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        BALBAK determines eigenvectors by undoing the BALANC transformation.

      Discussion:

        BALBAK forms the eigenvectors of a real general matrix by
        back transforming those of the corresponding balanced matrix
        determined by BALANC.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        15 July 2013

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Parlett and Reinsch,
        Numerische Mathematik,
        Volume 13, pages 293-304, 1969.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, column indices determined by BALANC.

        Input, double SCALE[N], contains information determining
        the permutations and scaling factors used by BALANC.

        Input, int M, the number of columns of Z to be
        back-transformed.

        Input/output, double Z[N*M], contains the real and imaginary
        parts of the eigenvectors, which, on return, have been back-transformed.
    */
    {
      int i;
      int ii;
      int j;
      int k;
      //double s;
      double t;

      if ( m <= 0 )
      {
        return;
      }

      if ( igh != low )
      {
        for ( i = low - 1; i <= igh - 1; i++ )
        {
          for ( j = 0; j < m; j++ )
          {
            z[i+j*n] = z[i+j*n] * scale[i];
          }
        }
      }

      for ( ii = 1; ii <= n; ii++ )
      {
        i = ii;

        if ( i < low || igh < i )
        {
          if ( i < low )
          {
            i = low - ii;
          }

          k = ( int ) ( scale[i-1] );

          if ( k != i )
          {
            for ( j = 0; j < m; j++ )
            {
              t          = z[i-1+j*n];
              z[i-1+j*n] = z[k-1+j*n];
              z[k-1+j*n] = t;
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void bandr ( int n, int mb, double a[], double d[], double e[], double e2[],
      bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        BANDR reduces a symmetric band matrix to symmetric tridiagonal form.

      Discussion:

        BANDR reduces a real symmetric band matrix
        to a symmetric tridiagonal matrix using and optionally
        accumulating orthogonal similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        09 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int MB, is the (half) band width of the matrix,
        defined as the number of adjacent diagonals, including the principal
        diagonal, required to specify the non-zero portion of the
        lower triangle of the matrix.

        Input/output, double A[N*MB].  On input, contains the lower
        triangle of the symmetric band input matrix stored as an N by MB array.
        Its lowest subdiagonal is stored in the last N+1-MB positions of the first
        column, its next subdiagonal in the last N+2-MB positions of the second
        column, further subdiagonals similarly, and finally its principal diagonal
        in the N positions of the last column.  Contents of storages not part of
        the matrix are arbitrary.  On output, A has been destroyed, except for
        its last two columns which contain a copy of the tridiagonal matrix.

        Output, double D[N], the diagonal elements of the tridiagonal
        matrix.

        Output, double E[N], the subdiagonal elements of the tridiagonal
        matrix in E(2:N).  E(1) is set to zero.

        Output, double E2[N], contains the squares of the corresponding
        elements of E.  E2 may coincide with E if the squares are not needed.

        Input, bool MATZ, is true if the transformation matrix is
        to be accumulated, and false otherwise.

        Output, double Z[N*N], the orthogonal transformation matrix
        produced in the reduction if MATZ is true.  Otherwise, Z is
        not referenced.
    */
    {
      double b1;
      double b2;
      double c2;
      double dmin;
      double dminrt;
      double f1;
      double f2;
      double g;
      int i;
      int i1;
      int i2;
      int j;
      int j1;
      int j2;
      int jj;
      int k;
      int kr;
      int l;
      int m1;
      int maxl;
      int maxr;
      int mr;
      int r;
      int r1;
      double s2;
      double u;
      int ugl;

      dmin = r8_epsilon ( );
      dminrt = sqrt ( dmin );
    /*
      Initialize the diagonal scaling matrix.
    */
      for ( i = 0; i < n; i++ )
      {
        d[i] = 1.0;
      }

      if ( matz )
      {
        r8mat_identity ( n, z );
      }
    /*
      Is input matrix diagonal?
    */
      if ( mb == 1 )
      {
        for ( i = 0; i < n; i++ )
        {
          d[i] = a[i+(mb-1)*n];
          e[i] = 0.0;
          e2[i] = 0.0;
        }
        return;
      }

      m1 = mb - 1;

      if ( m1 != 1 )
      {
        for ( k = 1; k <= n - 2; k++ )
        {
          maxr = i4_min ( m1, n - k );
          for ( r1 = 2; r1 <= maxr; r1++ )
          {
            r = maxr + 2 - r1;
            kr = k + r;
            mr = mb - r;
            g = a[kr-1+(mr-1)*n];
            a[kr-2+0*n] = a[kr-2+mr*n];
            ugl = k;

            for ( j = kr; j <= n; j = j + m1 )
            {
              j1 = j - 1;
              j2 = j1 - 1;

              if ( g == 0.0 )
              {
                break;
              }

              b1 = a[j1-1+0*n] / g;
              b2 = b1 * d[j1-1] / d[j-1];
              s2 = 1.0 / ( 1.0 + b1 * b2 );

              if ( s2 < 0.5 )
              {
                b1 = g / a[j1-1+0*n];
                b2 = b1 * d[j-1] / d[j1-1];
                c2 = 1.0 - s2;
                d[j1-1] = c2 * d[j1-1];
                d[j-1] = c2 * d[j-1];
                f1 = 2.0 * a[j-1+(m1-1)*n];
                f2 = b1 * a[j1-1+(mb-1)*n];
                a[j-1+(m1-1)*n] = - b2 * ( b1 * a[j-1+(m1-1)*n] - a[j-1+(mb-1)*n] )
                  - f2 + a[j-1+(m1-1)*n];
                a[j1-1+(mb-1)*n] = b2 * ( b2 * a[j-1+(mb-1)*n] + f1 ) + a[j1-1+(mb-1)*n];
                a[j-1+(mb-1)*n] = b1 * ( f2 - f1 ) + a[j-1+(mb-1)*n];

                for ( l = ugl; l <= j2; l++ )
                {
                  i2 = mb - j + l;
                  u = a[j1-1+i2*n] + b2 * a[j-1+(i2-1)*n];
                  a[j-1+(i2-1)*n] = -b1 * a[j1-1+i2*n] + a[j-1+(i2-1)*n];
                  a[j1-1+i2*n] = u;
                }

                ugl = j;
                a[j1-1+0*n] = a[j1+0*n] + b2 * g;

                if ( j != n )
                {
                  maxl = i4_min ( m1, n - j1 );

                  for ( l = 2; l <= maxl; l++ )
                  {
                    i1 = j1 + l;
                    i2 = mb - l;
                    u = a[i1-1+(i2-1)*n] + b2 * a[i1-1+i2*n];
                    a[i1-1+i2*n] = -b1 * a[i1-1+(i2-1)*n] + a[i1-1+i2*n];
                    a[i1-1+(i2-1)*n] = u;
                  }

                  i1 = j + m1;

                  if ( i1 <= n )
                  {
                    g = b2 * a[i1-1+0*n];
                  }
                }

                if ( matz )
                {
                  for ( l = 1; l <= n; l++ )
                  {
                    u = z[l-1+(j1-1)*n] + b2 * z[l-1+(j-1)*n];
                    z[l-1+(j-1)*n] = -b1 * z[l-1+(j1-1)*n] + z[l-1+(j-1)*n];
                    z[l-1+(j1-1)*n] = u;
                  }
                }
              }
              else
              {
                u = d[j1-1];
                d[j1-1] = s2 * d[j-1];
                d[j-1] = s2 * u;
                f1 = 2.0 * a[j-1+(m1-1)*n];
                f2 = b1 * a[j-1+(mb-1)*n];
                u = b1 * ( f2 - f1 ) + a[j1-1+(mb-1)*n];
                a[j-1+(m1-1)*n] = b2 * ( b1 * a[j-1+(m1-1)*n] - a[j1-1+(mb-1)*n] )
                  + f2 - a[j-1+(m1-1)*n];
                a[j1-1+(mb-1)*n] = b2 * ( b2 * a[j1-1+(mb-1)*n] + f1 )
                  + a[j-1+(mb-1)*n];
                a[j-1+(mb-1)*n] = u;

                for ( l = ugl; l <= j2; l++ )
                {
                  i2 = mb - j + l;
                  u = b2 * a[j1-1+i2*n] + a[j-1+(i2-1)*n];
                  a[j-1+(i2-1)*n] = -a[j1-1+i2*n] + b1 * a[j-1+(i2-1)*n];
                  a[j1-1+i2*n] = u;
                }

                ugl = j;
                a[j1-1+0*n] = b2 * a[j1-1+0*n] + g;

                if ( j != n )
                {
                  maxl = i4_min ( m1, n - j1 );

                  for ( l = 2; l <= maxl; l++ )
                  {
                    i1 = j1 + l;
                    i2 = mb - l;
                    u = b2 * a[i1-1+(i2-1)*n] + a[i1-1+i2*n];
                    a[i1-1+i2*n] = -a[i1-1+(i2-1)*n] + b1 * a[i1-1+i2*n];
                    a[i1-1+(i2-1)*n] = u;
                  }

                  i1 = j + m1;

                  if ( i1 <= n )
                  {
                    g = a[i1-1+0*n];
                    a[i1-1+0*n] = b1 * a[i1-1+0*n];
                  }
                }

                if ( matz )
                {
                  for ( l = 1; l <= n; l++ )
                  {
                    u = b2 * z[l-1+(j1-1)*n] + z[l-1+(j-1)*n];
                    z[l-1+(j-1)*n] = -z[l-1+(j1-1)*n] + b1 * z[l-1+(j-1)*n];
                    z[l-1+(j1-1)*n] = u;
                  }
                }
              }
            }
          }
    /*
      Rescale to avoid underflow or overflow.
    */
          if ( ( k % 64 ) == 0 )
          {
            for ( j = k; j <= n; j++ )
            {
              if ( d[j-1] < dmin )
              {
                maxl = i4_max ( 1, mb + 1 - j );

                for ( jj = maxl; jj <= m1; jj++ )
                {
                  a[j-1+(jj-1)*n] = dminrt * a[j-1+(jj-1)*n];
                }

                if ( j != n )
                {
                  maxl = i4_min ( m1, n - j );

                  for ( l = 1; l <= maxl; l++ )
                  {
                    i1 = j + l;
                    i2 = mb - l;
                    a[i1-1+(i2-1)*n] = dminrt * a[i1-1+(i2-1)*n];
                  }
                }

                if ( matz )
                {
                  for ( i = 1; i <= n; i++ )
                  {
                    z[i-1+(j-1)*n] = dminrt * z[i-1+(j-1)*n];
                  }
                }

                a[j-1+(mb-1)*n] = dmin * a[j-1+(mb-1)*n];
                d[j-1] = d[j-1] / dmin;
              }
            }
          }
        }
      }
    /*
      Form square root of scaling matrix.
    */
      for ( i = 1; i < n; i++ )
      {
        e[i] = sqrt ( d[i] );
      }
      if ( matz )
      {
        for ( j = 1; j < n; j++ )
        {
          for ( i = 0; i < n; i++ )
          {
            z[i+j*n] = z[i+j*n] * e[j];
          }
        }
      }

      u = 1.0;

      for ( j = 1; j < n; j++ )
      {
        a[j+(m1-1)*n] = u * e[j] * a[j+(m1-1)*n];
        u = e[j];
        e2[j] = a[j+(m1-1)*n] * a[j+(m1-1)*n];
        a[j+(mb-1)*n] = d[j] * a[j+(mb-1)*n];
        d[j] = a[j+(mb-1)*n];
        e[j] = a[j+(m1-1)*n];
      }

      d[0] = a[0+(mb-1)*n];
      e[0] = 0.0;
      e2[0] = 0.0;

      return;
    }
    /******************************************************************************/







    void cbabk2 ( int n, int low, int igh, double scale[], int m, double zr[],
      double zi[] )

    /******************************************************************************/
    /*
      Purpose:

        CBABK2 finds eigenvectors by undoing the CBAL transformation.

      Discussion:

        CBABK2 forms the eigenvectors of a complex general
        matrix by back transforming those of the corresponding
        balanced matrix determined by CBAL.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, values determined by CBAL.

        Input, double SCALE[N], information determining the permutations
        and scaling factors used by CBAL.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double ZR[N*M], ZI[N*M].  On input, the real
        and imaginary parts, respectively, of the eigenvectors to be back
        transformed in their first M columns.  On output, the transformed
        eigenvectors.
    */
    {
      int i;
      int ii;
      int j;
      int k;
      double s;

      if ( m == 0 )
      {
        return;
      }

      if ( igh != low )
      {
        for ( i = low; i <= igh; i++ )
        {
          s = scale[i];
          for ( j = 0; j < m; j++ )
          {
            zr[i+j*n] = zr[i+j*n] * s;
            zi[i+j*n] = zi[i+j*n] * s;
          }
        }
      }

      for ( ii = 0; ii < n; ii++ )
      {
        i = ii;
        if ( i < low || igh < i )
        {
          if ( i < low )
          {
            i = low - ii;
          }

          k = ( int ) scale[i];

          if ( k != i )
          {
            for ( j = 0; j < m; j++ )
            {
              s         = zr[i+j*n];
              zr[i+j*n] = zr[k+j*n];
              zr[k+j*n] = s;
              s         = zi[i+j*n];
              zi[i+j*n] = zi[k+j*n];
              zi[k+j*n] = s;
            }
          }
        }
      }
      return;
    }
    /******************************************************************************/


    void cdiv ( double ar, double ai, double br, double bi, double *cr, double *ci )

    /******************************************************************************/
    /*
      Purpose:

        CDIV emulates complex division, using real arithmetic.

      Discussion:

        CDIV performs complex division:

          (CR,CI) = (AR,AI) / (BR,BI)

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        18 October 2009

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        FORTRAN90 version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, double AR, AI, the real and imaginary parts of
        the numerator.

        Input, double BR, BI, the real and imaginary parts of
        the denominator.

        Output, double *CR, *CI, the real and imaginary parts of
        the result.
    */
    {
      double ais;
      double ars;
      double bis;
      double brs;
      double s;

      s = fabs ( br ) + fabs ( bi );

      ars = ar / s;
      ais = ai / s;
      brs = br / s;
      bis = bi / s;

      s = brs * brs + bis * bis;
      *cr = ( ars * brs + ais * bis ) / s;
      *ci = ( ais * brs - ars * bis ) / s;

      return;
    }
    /******************************************************************************/


    /******************************************************************************/

    int ch ( int n, double ar[], double ai[], double w[], bool matz, double zr[],
      double zi[] )

    /******************************************************************************/
    /*
      Purpose:

        CH gets eigenvalues and eigenvectors of a complex Hermitian matrix.

      Discussion:

        CH calls the recommended sequence of routines from the
        EISPACK eigensystem package to find the eigenvalues and eigenvectors
        of a complex hermitian matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        01 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double AR[N,N], AI[N,N].  On input, the real and
        imaginary parts of the complex matrix.  On output, AR and AI
        have been overwritten by other information.

        Output, double W[N], the eigenvalues in ascending order.

        Input, bool MATZ, is false if only eigenvalues are desired, and
        true if both eigenvalues and eigenvectors are to be computed.

        Output, double ZR[N,N], ZI[N,N], the real and imaginary parts,
        respectively, of the eigenvectors, if MATZ is true.

        Output, int CH, an error completion code described in
        the documentation for TQLRAT and TQL2.  The normal completion code is zero.
    */
    {
      double *fm1;
      double *fv1;
      double *fv2;
      int ierr;

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );
      fv2 = ( double * ) malloc ( n * sizeof ( double ) );
      fm1 = ( double * ) malloc ( 2 * n * sizeof ( double ) );

      htridi ( n, ar, ai, w, fv1, fv2, fm1 );

      if ( ! matz )
      {
        ierr = tqlrat ( n, w, fv2 );
        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          free ( fm1 );
          return ierr;
        }
      }
      else
      {
        r8mat_identity ( n, zr );

        ierr = tql2 ( n, w, fv1, zr );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          free ( fm1 );
          return ierr;
        }

        htribk ( n, ar, ai, fm1, n, zr, zi );
      }

      free ( fv1 );
      free ( fv2 );
      free ( fm1 );

      return ierr;
    }
    /******************************************************************************/

    int ch3 ( int n, double a[], double w[], bool matz, double zr[], double zi[] )

    /******************************************************************************/
    /*
      Purpose:

        CH3 gets eigenvalues and eigenvectors of a complex Hermitian matrix.

      Discussion:

        CH3 calls the recommended sequence of routines from the
        EISPACK eigensystem package to find the eigenvalues and eigenvectors
        of a complex hermitian matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        05 February 2018

      Author:

        John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double A[N,N].  On input, the lower triangle of
        the complex hermitian input matrix.  The real parts of the matrix elements
        are stored in the full lower triangle of A, and the imaginary parts are
        stored in the transposed positions of the strict upper triangle of A.  No
        storage is required for the zero imaginary parts of the diagonal elements.
        On output, A contains information about the unitary transformations
        used in the reduction.

        Output, double W[N], the eigenvalues in ascending order.

        Input, bool MATZ, is false if only eigenvalues are desired, and
        true if both eigenvalues and eigenvectors are to be computed.

        Output, double ZR[N,N], ZI[N,N], the real and imaginary parts,
        respectively, of the eigenvectors, if MATZ is true.

        Output, int CH, an error completion code described in
        the documentation for TQLRAT and TQL2.  The normal completion code is zero.
    */
    {
      double *fm1;
      double *fv1;
      double *fv2;
      int i;
      int ierr;
      int j;

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );
      fv2 = ( double * ) malloc ( n * sizeof ( double ) );
      fm1 = ( double * ) malloc ( 2 * n * sizeof ( double ) );

      htrid3 ( n, a, w, fv1, fv2, fm1 );

      if ( ! matz )
      {
        ierr = tqlrat ( n, w, fv2 );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          free ( fm1 );
          return ierr;
        }
      }
      else
      {
        r8mat_identity ( n, zr );

        for ( j = 0; j < n; j++ )
        {
          for ( i = 0; i < n; i++ )
          {
            zi[i+j*n] = 0.0;
          }
        }

        ierr = tql2 ( n, w, fv1, zr );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          free ( fm1 );
          return ierr;
        }

        htrib3 ( n, a, fm1, n, zr, zi );
      }

      free ( fv1 );
      free ( fv2 );
      free ( fm1 );

      return ierr;
    }
    /******************************************************************************/


    /******************************************************************************/

    void combak ( int n, int low, int igh, double ar[], double ai[], int inter[],
      int m, double zr[], double zi[] )

    /******************************************************************************/
    /*
      Purpose:

        COMBAK determines eigenvectors by undoing the COMHES transformation.

      Discussion:

        COMBAK forms the eigenvectors of a complex general
        matrix by back transforming those of the corresponding
        upper Hessenberg matrix determined by COMHES.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        24 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine CBAL.  If CBAL is not used, set LOW = 1 and IGH = to the order
        of the matrix.

        Input, double AR[N,IGH], AI[N,IGH], the multipliers which
        were used in the reduction by COMHES in their lower triangles below
        the subdiagonal.

        Input, int INTER[IGH], information on the rows and
        columns interchanged in the reduction by COMHES.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double ZR[N,M], ZI[N,M].  On input, the real
        and imaginary parts of the eigenvectors to be back transformed.  On
        output, the real and imaginary parts of the transformed eigenvectors.
    */
    {
      int i;
      int j;
      int mm;
      int mp;
      double t;
      double xi;
      double xr;

      if ( m == 0 )
      {
        return;
      }

      if ( igh - 1 < low + 1 )
      {
        return;
      }

      for ( mm = low + 1; mm <= igh - 1; mm++ )
      {
         mp = low + igh - mm;

         for ( i = mp + 1; i <= igh; i++ )
         {
            xr = ar[i+(mp-1)*n];
            xi = ai[i+(mp-1)*n];

            if ( xr != 0.0 || xi != 0.0 )
            {
              for ( j = 0; j < m; j++ )
              {
                zr[i+j*n] = zr[i+j*n] + xr * zr[mp+j*n] - xi * zi[mp+j*n];
                zi[i+j*n] = zi[i+j*n] + xr * zi[mp+j*n] + xi * zr[mp+j*n];
              }
           }
         }

         i = inter[mp];

         if ( i != mp )
         {
           for ( j = 0; j < m; j++ )
           {
             t          = zr[i+j*n];
             zr[i+j*n]  = zr[mp+j*n];
             zr[mp+j*n] = t;

             t          = zi[i+j*n];
             zi[i+j*n]  = zi[mp+j*n];
             zi[mp+j*n] = t;
           }
         }
      }

      return;
    }
    /******************************************************************************/

    void comhes ( int n, int low, int igh, double ar[], double ai[], int inter[] )

    /******************************************************************************/
    /*
      Purpose:

        COMHES transforms a complex general matrix to upper Hessenberg form.

      Discussion:

        COMHES is given a complex general matrix and reduces a submatrix in rows
        and columns LOW through IGH to upper Hessenberg form by
        stabilized elementary similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        04 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine CBAL.  If CBAL is not used, set LOW = 1 and IGH = N.

        Input/output, double AR[N,N], AI[N,N].  On input, the real and
        imaginary parts of the complex input matrix.  On output, the real and
        imaginary parts of the Hessenberg matrix.  The multipliers which were
        used in the reduction are stored in the remaining triangles under the
        Hessenberg matrix.

        Output, int INTER[IGH], information on the rows and
        columns interchanged in the reduction.
    */
    {
      int i;
      int j;
      int m;
      double t;
      double xi;
      double xr;
      double yi;
      double yr;

      for ( m = low + 1; m <= igh - 1; m++ )
      {
    /*
      Choose the pivot I.
    */
        xr = 0.0;
        xi = 0.0;
        i = m;

        for ( j = m; j <= igh; j++ )
        {
          if ( fabs ( xr )            + fabs ( xi ) <
               fabs ( ar[j+(m-1)*n] ) + fabs ( ai[j+(m-1)*n] ) )
          {
            xr = ar[j+(m-1)*n];
            xi = ai[j+(m-1)*n];
            i = j;
          }
        }

        inter[m] = i;
    /*
      Interchange rows and columns of AR and AI.
    */
        if ( i != m )
        {
          for ( j = m - 1; j < n; j++ )
          {
            t         = ar[i+j*n];
            ar[i+j*n] = ar[m+j*n];
            ar[m+j*n] = t;
            t         = ai[i+j*n];
            ai[i+j*n] = ai[m+j*n];
            ai[m+j*n] = t;
          }

          for ( j = 0; j <= igh; j++ )
          {
            t         = ar[j+i*n];
            ar[j+i*n] = ar[j+m*n];
            ar[j+m*n] = t;
            t         = ai[j+i*n];
            ai[j+i*n] = ai[j+m*n];
            ai[j+m*n] = t;
          }
        }
    /*
      Carry out the transformation.
    */
        if ( xr != 0.0 || xi != 0.0 )
        {
          for ( i = m + 1; i <= igh; i++ )
          {
            yr = ar[i+(m-1)*n];
            yi = ai[i+(m-1)*n];

            if ( yr != 0.0 || yi != 0.0 )
            {
              cdiv ( yr, yi, xr, xi, &yr, &yi );
              ar[i+(m-1)*n] = yr;
              ai[i+(m-1)*n] = yi;

              for ( j = m; j < n; j++ )
              {
                ar[i+j*n] = ar[i+j*n] - yr * ar[m+j*n] + yi * ai[m+j*n];
                ai[i+j*n] = ai[i+j*n] - yr * ai[m+j*n] - yi * ar[m+j*n];
              }

              for ( j = 0; j <= igh; j++ )
              {
                ar[j+m*n] = ar[j+m*n] + yr * ar[j+i*n] - yi * ai[j+i*n];
                ai[j+m*n] = ai[j+m*n] + yr * ai[j+i*n] + yi * ar[j+i*n];
              }
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    int comlr ( int n, int low, int igh, double hr[], double hi[], double wr[],
      double wi[] )

    /******************************************************************************/
    /*
      Purpose:

        COMLR gets all eigenvalues of a complex upper Hessenberg matrix.

      Discussion:

        COMLR finds the eigenvalues of a complex upper Hessenberg matrix by the
        modified LR method.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        10 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine CBAL.  If CBAL is not used, set LOW = 1 and IGH = N.

        Input/output, double HR[N,N], HI[N,N].  On input, the real and
        imaginary parts of the complex upper Hessenberg matrix.  Their lower
        triangles below the subdiagonal contain the multipliers which were used
        in the reduction by COMHES if performed.  On output, the upper Hessenberg
        portions of HR and HI have been destroyed.  Therefore, they must be
        saved before calling COMLR if subsequent calculation of eigenvectors
        is to be performed.

        Output, double WR[N], WI[N], the real and imaginary parts of the
        eigenvalues.  If an error break; is made, the eigenvalues should be correct
        for indices IERR+1,...,N.

        Output, int COMLR, error flag.
        0, for normal return,
        J, if the limit of 30*N iterations is exhausted while the J-th
          eigenvalue is being sought.
    */
    {
      double ai;
      double ar;
      int en;
      int i;
      int ierr;
      int itn;
      int its;
      int j;
      int l;
      int m;
      //int mm;
      double si;
      double sr;
      double t;
      double ti;
      double tr;
      double tst1;
      double tst2;
      double xi;
      double xr;
      double yi;
      double yr;
      double zzi;
      double zzr;

      ierr = 0;
    /*
      Store roots isolated by CBAL.
    */
      for ( i = 0; i < n; i++ )
      {
        if ( i < low || igh < i )
        {
          wr[i] = hr[i+i*n];
          wi[i] = hi[i+i*n];
        }
      }

      en = igh;
      tr = 0.0;
      ti = 0.0;
      itn = 30 * n;
    /*
      Search for next eigenvalue.
    */
      if ( en < low )
      {
        return ierr;
      }

      its = 0;
    /*
      Look for single small sub-diagonal element.
    */
      while ( true )
      {
        for ( l = en; low <= l; l-- )
        {
          if ( l == low )
          {
            break;
          }

          tst1 = fabs ( hr[l-1+(l-1)*n] ) + fabs ( hi[l-1+(l-1)*n] ) + fabs ( hr[l+l*n] )
            + fabs ( hi[l+l*n] );
          tst2 = tst1 + fabs ( hr[l+(l-1)*n] ) + fabs ( hi[l+(l-1)*n] );

          if ( tst2 == tst1 )
          {
            break;
          }
        }
    /*
      A root found.
    */
        if ( l == en )
        {
          wr[en] = hr[en+en*n] + tr;
          wi[en] = hi[en+en*n] + ti;
          en = en - 1;
          if ( en < low )
          {
            return ierr;
          }
          its = 0;
          continue;
        }

        if ( itn == 0 )
        {
          ierr = en;
          return ierr;
        }

        if ( its == 10 || its == 20 )
        {
          sr = fabs ( hr[en+(en-1)*n] ) + fabs ( hr[en-1+(en-2)*n] );
          si = fabs ( hi[en+(en-1)*n] ) + fabs ( hi[en-1+(en-2)*n] );
        }
        else
        {
          sr = hr[en+en*n];
          si = hi[en+en*n];
          xr = hr[en-1+en*n] * hr[en+(en-1)*n] - hi[en-1+en*n] * hi[en+(en-1)*n];
          xi = hr[en-1+en*n] * hi[en+(en-1)*n] + hi[en-1+en*n] * hr[en+(en-1)*n];

          if ( xr != 0.0 || xi != 0.0 )
          {
            yr = ( hr[en-1+(en-1)*n] - sr) / 2.0;
            yi = ( hi[en-1+(en-1)*n] - si) / 2.0;
            ar = yr * yr - yi * yi + xr;
            ai = 2.0 * yr * yi + xi;
            csroot ( ar, ai, &zzr, &zzi );

            if ( yr * zzr + yi * zzi < 0.0 )
            {
              zzr = - zzr;
              zzi = - zzi;
            }

            ar = yr + zzr;
            ai = yi + zzi;
            cdiv ( xr, xi, ar, ai, &xr, &xi );
            sr = sr - xr;
            si = si - xi;
          }
        }

        for ( i = low; i <= en; i++ )
        {
          hr[i+i*n] = hr[i+i*n] - sr;
          hi[i+i*n] = hi[i+i*n] - si;
        }

        tr = tr + sr;
        ti = ti + si;
        its = its + 1;
        itn = itn - 1;
    /*
      Look for two consecutive small sub-diagonal elements.
    */
        xr = fabs ( hr[en-1+(en-1)*n] ) + fabs ( hi[en-1+(en-1)*n] );
        yr = fabs ( hr[en+(en-1)*n] ) + fabs ( hi[en+(en-1)*n] );
        zzr = fabs ( hr[en+en*n] ) + fabs ( hi[en+en*n] );

        for ( m = en - 1; l <= m; m-- )
        {
          if ( m == l )
          {
            break;
          }

          yi = yr;
          yr = fabs ( hr[m+(m-1)*n] ) + fabs ( hi[m+(m-1)*n] );
          xi = zzr;
          zzr = xr;
          xr = fabs ( hr[m-1+(m-1)*n] ) + fabs ( hi[m-1+(m-1)*n] );
          tst1 = zzr / yi * ( zzr + xr + xi );
          tst2 = tst1 + yr;
          if ( tst2 == tst1 )
          {
            break;
          }
        }
    /*
      Triangular decomposition H=L*R.
    */
        for ( i = m + 1; i <= en; i++ )
        {
          xr = hr[i-1+(i-1)*n];
          xi = hi[i-1+(i-1)*n];
          yr = hr[i+(i-1)*n];
          yi = hi[i+(i-1)*n];

          if (  fabs ( yr ) + fabs ( yi ) <= fabs ( xr ) + fabs ( xi ) )
          {
            cdiv ( yr, yi, xr, xi, &zzr, &zzi );
            wr[i] = - 1.0;
          }
    /*
      Interchange rows of HR and HI.
    */
          else
          {
            for ( j = i - 1; j <= en; j++ )
            {
              t           = hr[i-1+j*n];
              hr[i-1+j*n] = hr[i+j*n];
              hr[i+j*n]   = t;
              t           = hi[i-1+j*n];
              hi[i-1+j*n] = hi[i+j*n];
              hi[i+j*n]   = t;
            }

            cdiv ( xr, xi, yr, yi, &zzr, &zzi );
            wr[i] = 1.0;
          }

          hr[i+(i-1)*n] = zzr;
          hi[i+(i-1)*n] = zzi;

          for ( j = i; j <= en; j++ )
          {
            hr[i+j*n] = hr[i+j*n] - zzr * hr[i-1+j*n] + zzi * hi[i-1+j*n];
            hi[i+j*n] = hi[i+j*n] - zzr * hi[i-1+j*n] - zzi * hr[i-1+j*n];
          }
        }
    /*
      Composition R*L=H.
    */
        for ( j = m + 1; j <= en; j++ )
        {
          xr = hr[j+(j-1)*n];
          xi = hi[j+(j-1)*n];
          hr[j+(j-1)*n] = 0.0;
          hi[j+(j-1)*n] = 0.0;
    /*
      Interchange columns of HR and HI, if necessary.
    */
          if ( 0.0 < wr[j] )
          {
            for ( i = l; i <= j; i++ )
            {
              t             = hr[i+(j-1)*n];
              hr[i+(j-1)*n] = hr[i+j*n];
              hr[i+j*n]     = t;
              t             = hi[i+(j-1)*n];
              hi[i+(j-1)*n] = hi[i+j*n];
              hi[i+j*n]     = t;
            }
          }

          for ( i = l; i <= j; i++ )
          {
            hr[i+(j-1)*n] = hr[i+(j-1)*n] + xr * hr[i+j*n] - xi * hi[i+j*n];
            hi[i+(j-1)*n] = hi[i+(j-1)*n] + xr * hi[i+j*n] + xi * hr[i+j*n];
          }
        }
      }

      return ierr;
    }
    /******************************************************************************/


    /******************************************************************************/

    int comqr ( int n, int low, int igh, double hr[], double hi[], double wr[],
      double wi[] )

    /******************************************************************************/
    /*
      Purpose:

        COMQR gets eigenvalues of a complex upper Hessenberg matrix.

      Discussion:

        COMQR finds the eigenvalues of a complex upper Hessenberg matrix by
        the QR method.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        11 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine CBAL.  If CBAL is not used, set LOW = 1 and IGH = N.

        Input/output, double HR[N,N], HI[N,N].  On input, the real
        and imaginary parts of the complex upper Hessenberg matrix.  Their lower
        triangles below the subdiagonal contain information about the unitary
        transformations used in the reduction by CORTH, if performed.  On output,
        the upper Hessenberg portions of HR and HI have been destroyed.
        Therefore, they must be saved before calling COMQR if subsequent
        calculation of eigenvectors is to be performed.

        Output, double WR[N], WI[N], the real and imaginary parts of the
        eigenvalues.  If an error break; is made, the eigenvalues should be
        correct for indices IERR+1,...,N.

        Output, int COMQR, error flag.
        0, for normal return,
        J, if the limit of 30*N iterations is exhausted while the J-th
           eigenvalue is being sought.
    */
    {
      double ai;
      double ar;
      int en;
      int enm1;
      int i;
      int ierr;
      int itn;
      int its;
      int j;
      int l;
      int ll;
      double norm;
      double si;
      double sr;
      double ti;
      double tr;
      double tst1;
      double tst2;
      double xi;
      double xr;
      double yi;
      double yr;
      double zzi;
      double zzr;

      ierr = 0;
    /*
      Create real subdiagonal elements.
    */
      l = low + 1;

      for ( i = low + 1; i <= igh; i++ )
      {
        ll = i4_min ( i + 1, igh );

        if ( hi[i+(i-1)*n] != 0.0 )
        {
          norm = pythag ( hr[i+(i-1)*n], hi[i+(i-1)*n] );
          yr = hr[i+(i-1)*n] / norm;
          yi = hi[i+(i-1)*n] / norm;
          hr[i+(i-1)*n] = norm;
          hi[i+(i-1)*n] = 0.0;

          for ( j = i; j <= igh; j++ )
          {
            si = yr * hi[i+j*n] - yi * hr[i+j*n];
            hr[i+j*n] = yr * hr[i+j*n] + yi * hi[i+j*n];
            hi[i+j*n] = si;
          }

          for ( j = low; j <= ll; j++ )
          {
            si = yr * hi[j+i*n] + yi * hr[j+i*n];
            hr[j+i*n] = yr * hr[j+i*n] - yi * hi[j+i*n];
            hi[j+i*n] = si;
          }
        }
      }
    /*
      Store roots isolated by CBAL.
    */
      for ( i = 0; i < n; i++ )
      {
        if ( i < low || igh < i )
        {
          wr[i] = hr[i+i*n];
          wi[i] = hi[i+i*n];
        }
      }

      en = igh;
      tr = 0.0;
      ti = 0.0;
      itn = 30 * n;
    /*
      Search for next eigenvalue.
    */
      if ( en < low )
      {
        return ierr;
      }

      its = 0;
      enm1 = en - 1;
    /*
      Look for single small sub-diagonal element.
    */
      while ( true )
      {
        for ( l = en; low <= l; l-- )
        {
          if ( l == low )
          {
            break;
          }
          tst1 = fabs ( hr[l-1+(l-1)*n] ) + fabs ( hi[l-1+(l-1)*n] ) + fabs ( hr[l+l*n] )
            + fabs ( hi[l+l*n] );
          tst2 = tst1 + fabs ( hr[l+(l-1)*n] );
          if ( tst2 == tst1 )
          {
            break;
          }
        }
    /*
      A root found.
    */
        if ( l == en )
        {
          wr[en] = hr[en+en*n] + tr;
          wi[en] = hi[en+en*n] + ti;
          en = enm1;
          if ( en < low )
          {
            break;
          }
          its = 0;
          enm1 = en - 1;
          continue;
        }

        if ( itn == 0 )
        {
          ierr = en;
          break;
        }

        if ( its == 10 || its == 20 )
        {
          sr = fabs ( hr[en+enm1*n] ) + fabs ( hr[enm1+(en-2)*n] );
          si = 0.0;
        }
        else
        {
          sr = hr[en+en*n];
          si = hi[en+en*n];
          xr = hr[enm1+en*n] * hr[en+enm1*n];
          xi = hi[enm1+en*n] * hr[en+enm1*n];

          if ( xr != 0.0 || xi != 0.0 )
          {
            yr = ( hr[enm1+enm1*n] - sr ) / 2.0;
            yi = ( hi[enm1+enm1*n] - si ) / 2.0;

            ar = yr * yr - yi * yi + xr;
            ai = 2.0 * yr * yi + xi;
            csroot ( ar, ai, &zzr, &zzi );

            if ( yr * zzr + yi * zzi < 0.0 )
            {
              zzr = - zzr;
              zzi = - zzi;
            }

            ar = yr + zzr;
            ai = yi + zzi;
            cdiv ( xr, xi, ar, ai, &xr, &xi );
            sr = sr - xr;
            si = si - xi;
          }
        }

        for ( i = low; i <= en; i++ )
        {
          hr[i+i*n] = hr[i+i*n] - sr;
          hi[i+i*n] = hi[i+i*n] - si;
        }

        tr = tr + sr;
        ti = ti + si;
        its = its + 1;
        itn = itn - 1;
    /*
      Reduce to triangle (rows).
    */
        for ( i = l + 1; i <= en; i++ )
        {
          sr = hr[i+(i-1)*n];
          hr[i+(i-1)*n] = 0.0;
          norm = pythag ( pythag ( hr[i-1+(i-1)*n], hi[i-1+(i-1)*n] ), sr );
          xr = hr[i-1+(i-1)*n] / norm;
          wr[i-1] = xr;
          xi = hi[i-1+(i-1)*n] / norm;
          wi[i-1] = xi;
          hr[i-1+(i-1)*n] = norm;
          hi[i-1+(i-1)*n] = 0.0;
          hi[i+(i-1)*n] = sr / norm;

          for ( j = 0; j <= en; j++ )
          {
            yr = hr[i-1+j*n];
            yi = hi[i-1+j*n];
            zzr = hr[i+j*n];
            zzi = hi[i+j*n];
            hr[i-1+j*n] = xr * yr + xi * yi + hi[i+(i-1)*n] * zzr;
            hi[i-1+j*n] = xr * yi - xi * yr + hi[i+(i-1)*n] * zzi;
            hr[i+j*n] = xr * zzr - xi * zzi - hi[i+(i-1)*n] * yr;
            hi[i+j*n] = xr * zzi + xi * zzr - hi[i+(i-1)*n] * yi;
          }
        }

        si = hi[en+en*n];

        if ( si != 0.0 )
        {
          norm = pythag ( hr[en+en*n], si );
          sr = hr[en+en*n] / norm;
          si = si / norm;
          hr[en+en*n] = norm;
          hi[en+en*n] = 0.0;
        }
    /*
      Inverse operation (columns).
    */
        for ( j = l + 1; j <= en; j++ )
        {
          xr = wr[j-1];
          xi = wi[j-1];

          for ( i = l; i <= j; i++ )
          {
            yr = hr[i+(j-1)*n];
            yi = 0.0;
            zzr = hr[i+j*n];
            zzi = hi[i+j*n];
            if ( i != j )
            {
              yi = hi[i+(j-1)*n];
              hi[i+(j-1)*n] = xr * yi + xi * yr + hi[j+(j-1)*n] * zzi;
            }
            hr[i+(j-1)*n] = xr * yr - xi * yi + hi[j+(j-1)*n] * zzr;
            hr[i+j*n] = xr * zzr + xi * zzi - hi[j+(j-1)*n] * yr;
            hi[i+j*n] = xr * zzi - xi * zzr - hi[j+(j-1)*n] * yi;
          }
        }

        if ( si != 0.0 )
        {
          for ( i = l; i <= en; i++ )
          {
            yr = hr[i+en*n];
            yi = hi[i+en*n];
            hr[i+en*n] = sr * yr - si * yi;
            hi[i+en*n] = sr * yi + si * yr;
          }
        }
      }

      return ierr;
    }
    /******************************************************************************/


    /******************************************************************************/


    void cortb ( int n, int low, int igh, double ar[], double ai[], double ortr[],
      double orti[], int m, double zr[], double zi[] )

    /******************************************************************************/
    /*
      Purpose:

        CORTB determines eigenvectors by undoing the CORTH transformation.

      Discussion:

        CORTB forms the eigenvectors of a complex general
        matrix by back transforming those of the corresponding
        upper Hessenberg matrix determined by CORTH.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        02 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine CBAL.  If CBAL is not used, set LOW = 1 and IGH to the order
        of the matrix.

        Input, double AR[N,IGH], AI[N,IGH], information about the
        unitary transformations used in the reduction by CORTH in their strict
        lower triangles.

        Input/output, double ORTR[IGH], ORTI[IGH].  On input, further
        information about the transformations used in the reduction by CORTH.  On
        output, ORTR and ORTI have been further altered.

        Input, int M, the number of columns of ZR and ZI to be
        back transformed.

        Input/output, double ZR[N,M], ZI[N,M].  On input, the real and
        imaginary parts of the eigenvectors to be back transformed.  On output,
        the real and imaginary parts of the transformed eigenvectors.
    */
    {
      double gi;
      double gr;
      double h;
      int i;
      double ii=0;
      double ir;
      int j;
      int mp;
      double ri;
      double rr;

      if ( m == 0 )
      {
        return;
      }

      if ( igh - 1 < low + 1 )
      {
        return;
      }

      for ( mp = igh - 1; low + 1 <= mp; mp-- )
      {
        if ( ar[mp+(mp-1)*n] != 0.0 || ai[mp+(mp-1)*n] != 0.0 )
        {
          h = ar[mp+(mp-1)*n] * ortr[mp] + ai[mp+(mp-1)*n] * orti[mp];
          for ( i = mp + 1; i <= igh; i++ )
          {
            ortr[i] = ar[i+(mp-1)*n];
            orti[i] = ai[i+(mp-1)*n];
          }
          for ( j = 0; j < m; j++ )
          {
            rr = 0.0;
            ri = 0.0;
            for ( i = mp; i <= igh; i++ )
            {
              rr = rr + ortr[i] * zr[i+j*n];
              ii = ii + orti[i] * zi[i+j*n];
            }
            gr = ( rr + ii ) / h;

            ri = 0.0;
            ir = 0.0;
            for ( i = mp; i <= igh; i++ )
            {
              ri = ri + ortr[i] * zi[i+j*n];
              ir = ir + orti[i] * zr[i+j*n];
            }
            gi = ( ri - ir ) / h;

            for ( i = mp; i <= igh; i++ )
            {
              zr[i+j*n] = zr[i+j*n] + gr * ortr[i] - gi * orti[i];
              zi[i+j*n] = zi[i+j*n] + gr * orti[i] + gi * ortr[i];
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void corth ( int n, int low, int igh, double ar[], double ai[], double ortr[],
      double orti[] )

    /******************************************************************************/
    /*
      Purpose:

        CORTH transforms a complex general matrix to upper Hessenberg form.

      Discussion:

        CORTH is given a complex general matrix and reduces a submatrix situated
        in rows and columns LOW through IGH to upper Hessenberg form by
        unitary similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        06 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        FORTRAN90 version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine CBAL.  If CBAL is not used, set LOW = 1 and IGH = N.

        Input/output, double AR[N,N], AI[N,N].  On input, the real and
        imaginary parts of the complex input matrix.  On output, the real and
        imaginary parts of the Hessenberg matrix.  Information about the unitary
        transformations used in the reduction is stored in the remaining
        triangles under the Hessenberg matrix.

        Output, double ORTR[IGH], ORTI[IGH], further information about
        the transformations.
    */
    {
      double f;
      double fi;
      double fr;
      double g;
      double h;
      int i;
      int j;
      int m;
      double scale;

      if ( igh - 1 < low + 1 )
      {
        return;
      }

      for ( m = low + 1; m <= igh - 1; m++ )
      {
        h = 0.0;
        ortr[m] = 0.0;
        orti[m] = 0.0;
        scale = 0.0;
    /*
      Scale column.
    */
        for ( i = m; i<= igh; i++ )
        {
          scale = scale + fabs ( ar[i+(m-1)*n] ) + fabs ( ai[i+(m-1)*n] );
        }

        if ( scale == 0.0 )
        {
          continue;
        }

        for ( i = igh; i < m; i++ )
        {
          ortr[i] = ar[i+(m-1)*n] / scale;
          orti[i] = ai[i+(m-1)*n] / scale;
          h = h + ortr[i] * ortr[i] + orti[i] * orti[i];
        }

        g = sqrt ( h );
        f = pythag ( ortr[m], orti[m] );

        if ( f != 0.0 )
        {
          h = h + f * g;
          g = g / f;
          ortr[m] = ( 1.0 + g ) * ortr[m];
          orti[m] = ( 1.0 + g ) * orti[m];
        }
        else
        {
          ortr[m] = g;
          ar[m+(m-1)*n] = scale;
        }
    /*
      Form (I-(U*Ut)/h) * A.
    */
        for ( j = m; j < n; j++ )
        {
          fr = 0.0;
          fi = 0.0;
          for ( i = igh; m <= i; i-- )
          {
            fr = fr + ortr[i] * ar[i+j*n] + orti[i] * ai[i+j*n];
            fi = fi + ortr[i] * ai[i+j*n] - orti[i] * ar[i+j*n];
          }
          fr = fr / h;
          fi = fi / h;

          for ( i = m; i <= igh; i++ )
          {
            ar[i+j*n] = ar[i+j*n] - fr * ortr[i] + fi * orti[i];
            ai[i+j*n] = ai[i+j*n] - fr * orti[i] - fi * ortr[i];
          }
        }
    /*
      Form (I-(U*Ut)/h) * A * (I-(U*Ut)/h)
    */
        for ( i = 0; i <= igh; i++ )
        {
          fr = 0.0;
          fi = 0.0;
          for ( j = igh; m <= j; j-- )
          {
            fr = fr + ortr[j] * ar[i+j*n] - orti[j] * ai[i+j*n];
            fi = fi + ortr[j] * ai[i+j*n] + orti[j] * ar[i+j*n];
          }
          fr = fr / h;
          fi = fi / h;

          for ( j = m; j <= igh; j++ )
          {
            ar[i+j*n] = ar[i+j*n] - fr * ortr[j] - fi * orti[j];
            ai[i+j*n] = ai[i+j*n] + fr * orti[j] - fi * ortr[j];
          }
        }

        ortr[m] = scale * ortr[m];
        orti[m] = scale * orti[m];
        ar[m+(m-1)*n] = - g * ar[m+(m-1)*n];
        ai[m+(m-1)*n] = - g * ai[m+(m-1)*n];
      }

      return;
    }
    /******************************************************************************/

    void csroot ( double xr, double xi, double *yr, double *yi )

    /******************************************************************************/
    /*
      Purpose:

        CSROOT computes the complex square root of a complex quantity.

      Discussion:

        The branch of the square function is chosen so that
          0.0 <= YR
        and
          sign ( YI ) == sign ( XI )

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        12 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, double XR, XI, the real and imaginary parts of the
        quantity whose square root is desired.

        Output, double *YR, *YI, the real and imaginary parts of the
        square root.
    */
    {
      double s;
      double ti;
      double tr;

      tr = xr;
      ti = xi;
      s = sqrt ( 0.5 * ( pythag ( tr, ti ) + fabs ( tr ) ) );

      if ( 0.0 <= tr )
      {
        *yr = s;
      }

      if ( ti < 0.0 )
      {
        s = -s;
      }

      if ( tr <= 0.0 )
      {
        *yi = s;
      }

      if ( tr < 0.0 )
      {
        *yr = 0.5 * ( ti / *yi );
      }
      else if ( 0.0 < tr )
      {
        *yi = 0.5 * ( ti / *yr );
      }
      return;
    }
    /******************************************************************************/

    void elmbak ( int n, int low, int igh, double a[], int ind[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        ELMBAK determines eigenvectors by undoing the ELMHES transformation.

      Discussion:

        ELMBAK forms the eigenvectors of a real general
        matrix by back transforming those of the corresponding
        upper Hessenberg matrix determined by ELMHES.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        25 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, integers determined by the balancing
        routine BALANC.  If BALANC has not been used, set LOW = 1 and
        IGH equal to the order of the matrix.

        Input, double A[N,IGH], the multipliers which were used in the
        reduction by ELMHES in its lower triangle below the subdiagonal.

        Input, int IND[IGH]), information on the rows and columns
        interchanged in the reduction by ELMHES.

        Input, int M, the number of columns of Z to be back
        transformed.

        Input/output, double Z[N,M].  On input, the real and imaginary
        parts of the eigenvectors to be back transformed.  On output, the real and
        imaginary parts of the transformed eigenvectors.
    */
    {
      int i;
      int j;
      int la;
      int mm;
      int mp;
      double t;
      double x;

      if ( m == 0 )
      {
        return;
      }

      la = igh - 1;

      if ( la < low + 1 )
      {
        return;
      }

      for ( mm = low + 1; mm <= la; mm++ )
      {
        mp = low + igh - mm;

        for ( i = mp + 1; i < igh; i++ )
        {
          x = a[i+(mp-1)*n];
          if ( x != 0.0 )
          {
            for ( j = 0; j < m; j++ )
            {
              z[i+j*n] = z[i+j*n] + x * z[mp+j*n];
            }
          }
        }

        i = ind[mp];

        if ( i != mp )
        {
          for ( j = 0; j < m; j++ )
          {
            t         = z[i+j*n];
            z[i+j*n]  = z[mp+j*n];
            z[mp+j*n] = t;
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void elmhes ( int n, int low, int igh, double a[], int ind[] )

    /******************************************************************************/
    /*
      Purpose:

        ELMHES transforms a real general matrix to upper Hessenberg form.

      Discussion:

        ELMHES is given a real general matrix and reduces a submatrix
        situated in rows and columns LOW through IGH to upper Hessenberg
        form by stabilized elementary similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        29 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Martin, James Wilkinson,
        ELMHES,
        Numerische Mathematik,
        Volume 12, pages 349-368, 1968.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine BALANC.  If BALANC has not been used, set LOW = 1, IGH = N.

        Input/output, double A[N,N].  On input, the matrix to be
        reduced.  On output, the Hessenberg matrix.  The multipliers
        which were used in the reduction are stored in the
        remaining triangle under the Hessenberg matrix.

        Output, int IND[N], contains information on the rows
        and columns interchanged in the reduction.  Only elements LOW through
        IGH are used.
    */
    {
      int i;
      int j;
      int la;
      int m;
      double t;
      double x;
      double y;

      la = igh - 1;

      for ( m = low + 1; m <= la; m++ )
      {
        x = 0.0;
        i = m;

        for ( j = m; j <= igh; j++ )
        {
          if ( fabs ( x ) < fabs ( a[j+(m-1)*n] ) )
          {
            x = a[j+(m-1)*n];
            i = j;
          }
        }

        ind[m] = i;
    /*
      Interchange rows and columns of the matrix.
    */
        if ( i != m )
        {
          for ( j = m - 1; j < n; j++ )
          {
            t        = a[i+j*n];
            a[i+j*n] = a[m+j*n];
            a[m+j*n] = t;
          }

          for ( j = 0; j <= igh; j++ )
          {
            t        = a[j+i*n];
            a[j+i*n] = a[j+m*n];
            a[j+m*n] = t;
          }
        }

        if ( x != 0.0 )
        {
          for ( i = m + 1; i <= igh; i++ )
          {
            y = a[i+(m-1)*n];

            if ( y != 0.0 )
            {
              y = y / x;
              a[i+(m-1)*n] = y;

              for ( j = m; j < n; j++ )
              {
                a[i+j*n] = a[i+j*n] - y * a[m+j*n];
              }
              for ( j = 0; j <= igh; j++ )
              {
                a[j+m*n] = a[j+m*n] + y * a[j*i*n];
              }
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void eltran ( int n, int low, int igh, double a[], int ind[], double z[] )

    /******************************************************************************/
    /*
      Purpose:

        ELTRAN accumulates similarity transformations used by ELMHES.

      Discussion:

        ELTRAN accumulates the stabilized elementary
        similarity transformations used in the reduction of a
        real general matrix to upper Hessenberg form by ELMHES.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        29 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Peters, James Wilkinson,
        ELMTRANS,
        Numerische Mathematik,
        Volume 16, pages 181-204, 1970.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine BALANC.  If BALANC has not been used, set LOW = 1, IGH = N.

        Input, double A[N,IGH], the multipliers which were used in the
        reduction by ELMHES in its lower triangle below the subdiagonal.

        Input, int IND[IGH], information on the rows and columns
        interchanged in the reduction by ELMHES.

        Output, double Z[N,N], the transformation matrix produced in the
        reduction by ELMHES.
    */
    {
      int i;
      int j;
      int mp;
    /*
      Initialize Z to the identity matrix.
    */
      r8mat_identity ( n, z );

      if ( igh < low + 2 )
      {
        return;
      }

      for ( mp = igh - 1; low + 1 <= mp; mp-- )
      {
        for ( i = mp + 1; i <= igh; i++ )
        {
          z[i+mp*n] = a[i+(mp-1)*n];
        }
        i = ind[mp];

        if ( i != mp )
        {
          for ( j = mp; j <= igh; j++ )
          {
            z[mp+j*n] = z[i+j*n];
          }
          z[i+mp*n] = 1.0;
          for ( j = mp + 1; j <= igh; j++ )
          {
            z[i+j*n] = 0.0;
          }
        }
      }

      return;
    }
    /******************************************************************************/

    int figi ( int n, double t[], double d[], double e[], double e2[] )

    /******************************************************************************/
    /*
      Purpose:

        FIGI transforms a real nonsymmetric tridiagonal matrix to symmetric form.

      Discussion:

        FIGI is given a nonsymmetric tridiagonal matrix such that the products
        of corresponding pairs of off-diagonal elements are all
        non-negative.  It reduces the matrix to a symmetric
        tridiagonal matrix with the same eigenvalues.  If, further,
        a zero product only occurs when both factors are zero,
        the reduced matrix is similar to the original matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double T[N,3] contains the input matrix.  Its subdiagonal
        is stored in the last N-1 positions of the first column, its diagonal in
        the N positions of the second column, and its superdiagonal in the
        first N-1 positions of the third column.  T(1,1) and T(N,3) are arbitrary.

        Output, double D[N], the diagonal elements of the symmetric
        matrix.

        Output, double E[N], contains the subdiagonal elements of
        the symmetric matrix in E(2:N).  E(1) is not set.

        Output, double E2[N], the squares of the corresponding elements
        of E.  E2 may coincide with E if the squares are not needed.

        Output, int FIGI, error flag.
        0, for normal return,
        N+I, if T(I,1) * T(I-1,3) is negative,
        -(3*N+I), if T(I,1) * T(I-1,3) is zero with one factor non-zero.  In
          this case, the eigenvectors of the symmetric matrix are not simply
          related to those of T and should not be sought.
    */
    {
      int i;
      int ierr;

      ierr = 0;

      for ( i = 0; i < n; i++ )
      {
        if ( 0 < i )
        {
          e2[i] = t[i+0*n] * t[i-1+2*n];

          if ( e2[i] < 0.0 )
          {
            ierr = n + i + 1;
            return ierr;
          }
          else if ( e2[i] == 0.0 )
          {
            if ( t[i+0*n] != 0.0 || t[i-1+2*n] != 0.0 )
            {
              ierr = - 3 * n - i - 1;
              return ierr;
            }
            e[i] = 0.0;
          }
          else
          {
            e[i] = sqrt ( e2[i] );
          }
        }
        else
        {
          e[i] = 0.0;
          e2[i] = 0.0;
        }
        d[i] = t[i+1*n];
      }

      return ierr;
    }
    /******************************************************************************/

    int figi2 ( int n, double t[], double d[], double e[], double z[] )

    /******************************************************************************/
    /*
      Purpose:

        FIGI2 transforms a real nonsymmetric tridiagonal matrix to symmetric form.

      Discussion:

        FIGI2 is given a nonsymmetric tridiagonal matrix such that the products
        of corresponding pairs of off-diagonal elements are all
        non-negative, and zero only when both factors are zero.

        It reduces the matrix to a symmetric tridiagonal matrix
        using and accumulating diagonal similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double T[N,3] contains the input matrix.  Its subdiagonal
        is stored in the last N-1 positions of the first column, its diagonal in
        the N positions of the second column, and its superdiagonal in the
        first N-1 positions of the third column.  T(1,1) and T(N,3) are arbitrary.

        Output, double D[N], the diagonal elements of the symmetric
        matrix.

        Output, double E[N], contains the subdiagonal elements of the
        symmetric matrix in E(2:N).  E(1) is not set.

        Output, double Z[N,N], contains the transformation matrix
        produced in the reduction.

        Output, int FIGI2, error flag.
        0, for normal return,
        N+I, if T(I,1) * T(I-1,3) is negative,
        2*N+I, if T(I,1) * T(I-1,3) is zero with one factor non-zero.
    */
    {
      double h;
      int i;
      int ierr;
      //int j;

      ierr = 0;
    /*
      Initialize Z to the identity matrix.
    */
      r8mat_identity ( n, z );

      for ( i = 0; i < n; i++ )
      {
        if ( i == 0 )
        {
          e[i] = 0.0;
        }
        else
        {
          h = t[i+0*n] * t[i-1+2*n];

          if ( h < 0.0 )
          {
            ierr = n + i + 1;
            return ierr;
          }
          else if ( h == 0 )
          {
            if ( t[i+0*n] != 0.0 || t[i-1+2*n] != 0.0 )
            {
              ierr = 2 * n + i + 1;
              return ierr;
            }

            e[i] = 0.0;
            z[i+i*n] = 1.0;
          }
          else
          {
            e[i] = sqrt ( h );
            z[i+i*n] = z[i-1+(i-1)*n] * e[i] / t[i-1+2*n];
          }
        }
        d[i] = t[i+1*n];
      }

      return ierr;
    }
    /******************************************************************************/

    int hqr ( int n, int low, int igh, double h[], double wr[], double wi[] )

    /******************************************************************************/
    /*
      Purpose:

        HQR computes all eigenvalues of a real upper Hessenberg matrix.

      Discussion:

        HQR finds the eigenvalues of a real
        upper Hessenberg matrix by the QR method.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        10 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Martin, Peters, James Wilkinson,
        HQR,
        Numerische Mathematik,
        Volume 14, pages 219-231, 1970.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, two integers determined by
        BALANC.  If BALANC is not used, set LOW=1, IGH=N.

        Input/output, double H[N,N], the N by N upper Hessenberg matrix.
        Information about the transformations used in the reduction to
        Hessenberg form by ELMHES or ORTHES, if performed, is stored
        in the remaining triangle under the Hessenberg matrix.
        On output, the information in H has been destroyed.

        Output, double WR[N], WI[N], the real and imaginary parts of the
        eigenvalues.  The eigenvalues are unordered, except that complex
        conjugate pairs of values appear consecutively, with the eigenvalue
        having positive imaginary part listed first.  If an error break;
        occurred, then the eigenvalues should be correct for indices
        IERR+1 through N.

        Output, int HQR, error flag.
        0, no error.
        J, the limit of 30*N iterations was reached while searching for
          the J-th eigenvalue.
    */
    {
      int en;
      int enm2;
      int i;
      int ierr;
      int itn;
      int its;
      int j;
      int k;
      int l;
      //int ll;
      int m;
      //int mm;
      int na;
      double norm;
      bool notlas;
      double p = 0.0;
      double q = 0.0;
      double r = 0.0;
      double s;
      double t;
      double tst1;
      double tst2;
      double w;
      double x;
      double y;
      double zz;

      ierr = 0;
      norm = 0.0;
      k = 0;
    /*
      Store roots isolated by BALANC and compute matrix norm.
    */
      for ( i = 0; i < n; i++ )
      {
        for ( j = k; j < n; j++ )
        {
          norm = norm + fabs ( h[i+j*n] );
        }

        k = i;
        if ( i < low || igh < i )
        {
          wr[i] = h[i+i*n];
          wi[i] = 0.0;
        }
      }

      en = igh;
      t = 0.0;
      itn = 30 * n;
    /*
      Search for next eigenvalues.
    */
      if ( igh < low )
      {
        return ierr;
      }

      its = 0;
      na = igh - 1;
      enm2 = igh - 2;
    /*
      Look for a single small sub-diagonal element.
    */
      while ( true )
      {
        for ( l = en; low <= l; l-- )
        {
          if ( l == low )
          {
            break;
          }
          s = fabs ( h[l-1+(l-1)*n] ) + fabs ( h[l+l*n] );
          if ( s == 0.0 )
          {
            s = norm;
          }
          tst1 = s;
          tst2 = tst1 + fabs ( h[l+(l-1)*n] );
          if ( tst2 == tst1 )
          {
            break;
          }
        }
    /*
      Form shift.
    */
        x = h[en+en*n];
    /*
      One root found.
    */
        if ( l == en )
        {
          wr[en] = x + t;
          wi[en] = 0.0;
          en = na;
          if ( en < low )
          {
            return ierr;
          }
          its = 0;
          na = en - 1;
          enm2 = na - 1;
          continue;
        }

        y = h[na+na*n];
        w = h[en+na*n] * h[na+en*n];
    /*
      Two roots found.
    */
        if ( l == na )
        {
          p = ( y - x ) / 2.0;
          q = p * p + w;
          zz = sqrt ( fabs ( q ) );
          x = x + t;
    /*
      Real root, or complex pair.
    */
          if ( 0.0 <= q )
          {
            zz = p + fabs ( zz ) * r8_sign ( p );
            wr[na] = x + zz;
            if ( zz == 0.0 )
            {
              wr[en] = wr[na];
            }
            else
            {
              wr[en] = x - w / zz;
            }
            wi[na] = 0.0;
            wi[en] = 0.0;
          }
          else
          {
            wr[na] = x + p;
            wr[en] = x + p;
            wi[na] = zz;
            wi[en] = - zz;
          }

          en = enm2;

          if ( en < low )
          {
            return ierr;
          }

          its = 0;
          na = en - 1;
          enm2 = na - 1;
          continue;
        }

        if ( itn == 0 )
        {
          ierr = en;
          return ierr;
        }
    /*
      Form an exceptional shift.
    */
        if ( its == 10 || its == 20 )
        {
          t = t + x;

          for ( i = low; i <= en; i++ )
          {
            h[i+i*n] = h[i+i*n] - x;
          }

          s = fabs ( h[en+na*n] ) + fabs ( h[na+enm2*n] );
          x = 0.75 * s;
          y = x;
          w = - 0.4375 * s * s;
        }

        its = its + 1;
        itn = itn - 1;
    /*
      Look for two consecutive small sub-diagonal elements.
    */
        for ( m = enm2; l <= m; m-- )
        {
          zz = h[m+m*n];
          r = x - zz;
          s = y - zz;
          p = ( r * s - w ) / h[m+1+m*n] + h[m+(m+1)*n];
          q = h[m+1+(m+1)*n] - zz - r - s;
          r = h[m+2+(m+1)*n];
          s = fabs ( p ) + fabs ( q ) + fabs ( r );
          p = p / s;
          q = q / s;
          r = r / s;

          if ( m == l )
          {
            break;
          }

          tst1 = fabs ( p ) * ( fabs ( h[m-1+(m-1)*n] ) + fabs ( zz )
            + fabs ( h[m+1+(m+1)*n] ) );
          tst2 = tst1 + fabs ( h[m+(m-1)*n] ) * ( fabs ( q ) + fabs ( r ) );

          if ( tst2 == tst1 )
          {
            break;
          }
        }

        for ( i = m + 2; i <= en; i++ )
        {
          h[i+(i-2)*n] = 0.0;
          if ( i != m + 2 )
          {
            h[i+(i-3)*n] = 0.0;
          }
        }
    /*
      Double QR step involving rows l to EN and columns M to EN.
    */
        for ( k = m; k <= na; k++ )
        {
          notlas = ( k != na );

          if ( k != m )
          {
            p = h[k+(k-1)*n];
            q = h[k+1+(k-1)*n];

            if ( notlas )
            {
              r = h[k+2+(k-1)*n];
            }
            else
            {
              r = 0.0;
            }

            x = fabs ( p ) + fabs ( q ) + fabs ( r );

            if ( x == 0.0 )
            {
              continue;
            }
            p = p / x;
            q = q / x;
            r = r / x;
          }

          s = sqrt ( p * p + q * q + r * r ) * r8_sign ( p );

          if ( k != m )
          {
            h[k+(k-1)*n] = - s * x;
          }
          else if ( l != m )
          {
            h[k+(k-1)*n] = - h[k+(k-1)*n];
          }

          p = p + s;
          x = p / s;
          y = q / s;
          zz = r / s;
          q = q / p;
          r = r / p;
    /*
      Row modification.
    */
          if ( ! notlas )
          {
            for ( j = k; j < n; j++ )
            {
              p = h[k+j*n] + q * h[k+1+j*n];
              h[k+j*n] = h[k+j*n] - p * x;
              h[k+1+j*n] = h[k+1+j*n] - p * y;
            }

            j = i4_min ( en, k + 3 );
    /*
      Column modification.
    */
            for ( i = 0; i <= j; i++ )
            {
              p = x * h[i+k*n] + y * h[i+(k+1)*n];
              h[i+k*n] = h[i+k*n] - p;
              h[i+(k+1)*n] = h[i+(k+1)*n] - p * q;
            }
          }
    /*
      Row modification.
    */
          else
          {
            for ( j = k; j < n; j++ )
            {
              p = h[k+j*n] + q * h[k+1+j*n] + r * h[k+2+j*n];
              h[k+j*n] = h[k+j*n] - p * x;
              h[k+1+j*n] = h[k+1+j*n] - p * y;
              h[k+2+j*n] = h[k+2+j*n] - p * zz;
            }

            j = i4_min ( en, k + 3 );
    /*
      Column modification.
    */
            for ( i = 0; i <= j; i++ )
            {
              p = x * h[i+k*n] + y * h[i+(k+1)*n] + zz * h[i+(k+2)*n];
              h[i+k*n] = h[i+k*n] - p;
              h[i+(k+1)*n] = h[i+(k+1)*n] - p * q;
              h[i+(k+2)*n] = h[i+(k+2)*n] - p * r;
            }
          }
        }
      }

      return ierr;
    }
    /******************************************************************************/


    /******************************************************************************/

    void htrib3 ( int n, double a[], double tau[], int m, double zr[], double zi[] )

    /******************************************************************************/
    /*
      Purpose:

        HTRIB3 determines eigenvectors by undoing the HTRID3 transformation.

      Discussion:

        HTRIB3 forms the eigenvectors of a complex hermitian
        matrix by back transforming those of the corresponding
        real symmetric tridiagonal matrix determined by HTRID3.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        02 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, is the order of the matrix.

        Input, double A[N,N], contains information about the unitary
        transformations used in the reduction by HTRID3.

        Input, double TAU[2,N], contains further information about the
        transformations.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double ZR[N,M], ZI[N,M].  On input, ZR contains
        the eigenvectors to be back transformed.  On output, ZR and ZI contain
        the real and imaginary parts of the transformed eigenvectors.
    */
    {
      double h;
      int i;
      int j;
      int k;
      double s;
      double si;

      if ( m == 0 )
      {
        return;
      }
    /*
      Transform the eigenvectors of the real symmetric tridiagonal matrix
      to those of the hermitian tridiagonal matrix.
    */
      for ( k = 0; k < n; k++ )
      {
        for ( j = 0; j < m; j++ )
        {
          zi[k+j*n] = - zr[k+j*n] * tau[1+k*2];
          zr[k+j*n] =   zr[k+j*n] * tau[0+k*2];
        }
      }
    /*
      Recover and apply the Householder matrices.
    */
      for ( i = 1; i < n; i++ )
      {
        h = a[i+i*n];

        if ( h != 0.0 )
        {
          for ( j = 0; j < m; j++ )
          {
            s = 0.0;
            si = 0.0;

            for ( k = 0; k < i; k++ )
            {
              s  = s  + a[i+k*n] * zr[k+j*n] - a[k+i*n] * zi[k+j*n];
              si = si + a[i+k*n] * zi[k+j*n] + a[k+i*n] * zr[k+j*n];
            }

            s = ( s / h ) / h;
            si = ( si / h ) / h;

            for ( k = 0; k < i; k++ )
            {
              zr[k+j*n] = zr[k+j*n] - s  * a[i+k*n] - si * a[k+i*n];
              zi[k+j*n] = zi[k+j*n] - si * a[i+k*n] + s  * a[k+i*n];
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void htribk ( int n, double ar[], double ai[], double tau[], int m, double zr[],
      double zi[] )

    /******************************************************************************/
    /*
      Purpose:

        HTRIBK determines eigenvectors by undoing the HTRIDI transformation.

      Discussion:

        HTRIBK forms the eigenvectors of a complex hermitian
        matrix by back transforming those of the corresponding
        real symmetric tridiagonal matrix determined by HTRIDI.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        01 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double AR[N,N], AI[N,N], contain information about
        the unitary transformations used in the reduction by HTRIDI in their
        full lower triangles, except for the diagonal of AR.

        Input, double TAU[2,N], contains further information about the
        transformations.

        Input, int  M, the number of eigenvectors to be back
        transformed.

        Input/output, double ZR[N,M], ZI[N,M].  On input, ZR contains
        the eigenvectors to be back transformed.  On output, ZR and ZI contain
        the real and imaginary parts of the transformed eigenvectors.
    */
    {
      double h;
      int i;
      int j;
      int k;
      double s;
      double si;

      if ( m == 0 )
      {
        return;
      }
    /*
      Transform the eigenvectors of the real symmetric tridiagonal matrix to
      those of the hermitian tridiagonal matrix.
    */
      for ( k = 0; k < n; k++ )
      {
        for ( j = 0; j < m; j++ )
        {
          zi[k+j*n] = - zr[k+j*n] * tau[1+k*2];
          zr[k+j*n] =   zr[k+j*n] * tau[0+k*2];
        }
      }
    /*
      Recover and apply the Householder matrices.
    */
      for ( i = 1; i < n; i++ )
      {
        h = ai[i+i*n];

        if ( h != 0.0 )
        {
          for ( j = 0; j < m; j++ )
          {
            s = 0.0;
            si = 0.0;
            for ( k = 0; k < i; k++ )
            {
              s =  s  + ar[i+k*n] * zr[k+j*n] - ai[i+k*n] * zi[k+j*n];
              si = si + ar[i+k*n] * zi[k+j*n] + ai[i+k*n] * zr[k+j*n];
            }

            s = ( s / h ) / h;
            si = ( si / h ) / h;

            for ( k = 0; k < i; k++ )
            {
              zr[k+j*n] = zr[k+j*n] - s  * ar[i+k*n] - si * ai[i+k*n];
              zi[k+j*n] = zi[k+j*n] - si * ar[i+k*n] + s  * ai[i+k*n];
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void htrid3 ( int n, double a[], double d[], double e[], double e2[],
      double tau[] )

    /******************************************************************************/
    /*
      Purpose:

        HTRID3 tridiagonalizes a complex hermitian packed matrix.

      Discussion:

        HTRID3 reduces a complex hermitian matrix, stored as a single square
        array, to a real symmetric tridiagonal matrix using unitary similarity
        transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double A[N,N].  On input, the lower triangle of
        the complex hermitian input matrix.  The real parts of the matrix elements
        are stored in the full lower triangle of A, and the imaginary parts are
        stored in the transposed positions of the strict upper triangle of A.  No
        storage is required for the zero imaginary parts of the diagonal elements.
        On output, A contains information about the unitary transformations
        used in the reduction.

        Output, double D[N], the diagonal elements of the
        tridiagonal matrix.

        Output, double E[N], the subdiagonal elements of the tridiagonal
        matrix in E(2:N).  E(1) is set to zero.

        Output, double E2[N], the squares of the corresponding elements
        of E.  E2 may coincide with E if the squares are not needed.

        Output, double TAU[2,N], contains further information about the
        transformations.
    */
    {
      double f;
      double fi;
      double g;
      double gi;
      double h;
      double hh;
      int i;
      int j;
      int k;
      double scale;
      double si;

      tau[0+(n-1)*2] = 1.0;
      tau[1+(n-1)*2] = 0.0;

      for ( i = n - 1; 0 <= i; i-- )
      {
        h = 0.0;
        scale = 0.0;

        if ( i < 1 )
        {
          e[i] = 0.0;
          e2[i] = 0.0;
        }
        else
        {
    /*
      Scale row.
    */
          for ( k = 0; k < i; k++ )
          {
            scale = scale + fabs ( a[i+k*n] ) + fabs ( a[k+i*n] );
          }

          if ( scale == 0.0 )
          {
            tau[0+(i-1)*2] = 1.0;
            tau[1+(i-1)*2] = 0.0;
            e[i] = 0.0;
            e2[i] = 0.0;
          }
          else
          {
            for ( k = 0; k < i; k++ )
            {
              a[i+k*n] = a[i+k*n] / scale;
              a[k+i*n] = a[k+i*n] / scale;
              h = h + a[i+k*n] * a[i+k*n] + a[k+i*n] * a[k+i*n];
            }

            e2[i] = scale * scale * h;
            g = sqrt ( h );
            e[i] = scale * g;
            f = pythag ( a[i+(i-1)*n], a[i-1+i*n] );
    /*
      Form next diagonal element of matrix T.
    */
            if ( f != 0.0 )
            {
              tau[0+(i-1)*2] = ( a[i-1+i*n]   * tau[1+i*2] - a[i+(i-1)*n] * tau[0+i*2] ) / f;
              si =             ( a[i+(i-1)*n] * tau[1+i*2] + a[i-1+i*n]   * tau[0+i*2] ) / f;
              h = h + f * g;
              g = 1.0 + g / f;
              a[i+(i-1)*n] = g * a[i+(i-1)*n];
              a[i-1+i*n]   = g * a[i-1+i*n];

              if ( i == 1 )
              {
                a[1+0*n] = scale * a[1+0*n];
                a[0+1*n] = scale * a[0+1*n];
                tau[1+(i-1)*2] = - si;
                d[i] = a[i+i*n];
                a[i+i*n] = scale * sqrt ( h );
                continue;
              }
            }
            else
            {
              tau[0+(i-1)*2] = - tau[0+i*2];
              si = tau[1+i*2];
              a[i+(i-1)*n] = g;
            }

            f = 0.0;

            for ( j = 0; j < i; j++ )
            {
              g = 0.0;
              gi = 0.0;
    /*
      Form element of A*U.
    */
              for ( k = 0; k < j; k++ )
              {
                g  = g  + a[j+k*n] * a[i+k*n] + a[k+j*n] * a[k+i*n];
                gi = gi - a[j+k*n] * a[k+i*n] + a[k+j*n] * a[i+k*n];
              }

              g  = g  + a[j+j*n] * a[i+j*n];
              gi = gi - a[j+j*n] * a[j+i*n];

              for ( k = j + 1; k < i; k++ )
              {
                g  = g  + a[k+j*n] * a[i+k*n] - a[j+k*n] * a[k+i*n];
                gi = gi - a[k+j*n] * a[k+i*n] - a[j+k*n] * a[i+k*n];
              }
    /*
      Form element of P.
    */
              e[j] = g / h;
              tau[1+j*2] = gi / h;
              f = f + e[j] * a[i+j*n] - tau[1+j*2] * a[j+i*n];
            }

            hh = f / ( h + h );
    /*
      Form reduced A.
    */
            for ( j = 0; j < i; j++ )
            {
              f = a[i+j*n];
              g = e[j] - hh * f;
              e[j] = g;
              fi = - a[j+i*n];
              gi = tau[1+j*2] - hh * fi;
              tau[1+j*2] = - gi;
              a[j+j*n] = a[j+j*n] - 2.0 * ( f * g + fi * gi );

              for ( k = 0; k < j; k++ )
              {
                a[j+k*n] = a[j+k*n]
                  - f * e[k] - g * a[i+k*n] + fi * tau[1+k*2] + gi * a[k+i*n];
                a[k+j*n] = a[k+j*n]
                  - f * tau[1+k*2] - g * a[k+i*n] - fi * e[k] - gi * a[i+k*n];
              }
            }

            for ( j = 0; j < i; j++ )
            {
              a[i+j*n] = scale * a[i+j*n];
              a[j+i*n] = scale * a[j+i*n];
            }
            tau[1+(i-1)*2] = - si;
          }
        }

        d[i] = a[i+i*n];
        a[i+i*n] = scale * sqrt ( h );
      }

      return;
    }
    /******************************************************************************/

    void htridi ( int n, double ar[], double ai[], double d[], double e[],
      double e2[], double tau[] )

    /******************************************************************************/
    /*
      Purpose:

        HTRIDI tridiagonalizes a complex hermitian matrix.

      Discussion:

        HTRIDI reduces a complex hermitian matrix to a real symmetric
        tridiagonal matrix using unitary similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        01 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double AR[N,N], AI[N,N].  On input, the real
        and imaginary parts, respectively, of the complex hermitian input matrix.
        Only the lower triangle of the matrix need be supplied.
        On output, information about the unitary transformations used in the
        reduction in their full lower triangles.  Their strict upper triangles
        and the diagonal of AR are unaltered.

        Output, double D[N], the diagonal elements of the
        tridiagonal matrix.

        Output, double E[N], the subdiagonal elements of the tridiagonal
        matrix in its last N-1 positions.  E(1) is set to zero.

        Output, double E2[N], the squares of the corresponding elements
        of E.  E2 may coincide with E if the squares are not needed.

        Output, double TAU[2,N], contains further information about the
        transformations.
    */
    {
      double f;
      double fi;
      double g;
      double gi;
      double h;
      double hh;
      int i;
      //int ii;
      int j;
      int k;
      int l;
      double scale;
      double si;

      tau[0+(n-1)*2] = 1.0;
      tau[1+(n-1)*2] = 0.0;

      for ( i = 0; i < n; i++ )
      {
        d[i] = ar[i+i*n];
      }

      for ( i = n - 1; 0 <= i; i-- )
      {
        l = i - 1;
        h = 0.0;
        scale = 0.0;

        if ( i == 0 )
        {
          e[i] = 0.0;
          e2[i] = 0.0;
          hh = d[i];
          d[i] = ar[i+i*n];
          ar[i+i*n] = hh;
          ai[i+i*n] = scale * sqrt ( h );
        }
    /*
      Scale row.
    */
        else
        {
          for ( k = 0; k < i; k++ )
          {
            scale = scale + fabs ( ar[i+k*n] ) + fabs ( ai[i+k*n] );
          }

          if ( scale == 0.0 )
          {
            tau[0+(i-1)*2] = 1.0;
            tau[1+(i-1)*2] = 0.0;
            e[i] = 0.0;
            e2[i] = 0.0;
            hh = d[i];
            d[i] = ar[i+i*n];
            ar[i+i*n] = hh;
            ai[i+i*n] = scale * sqrt ( h );
            continue;
          }

          for ( j = 0; j < i; j++ )
          {
            ar[i+j*n] = ar[i+j*n] / scale;
            ai[i+j*n] = ai[i+j*n] / scale;
          }

          for ( k = 0; k < i; k++ )
          {
            h = h + ar[i+k*n] * ar[i+k*n] + ai[i+k*n] * ai[i+k*n];
          }

          e2[i] = scale * scale * h;
          g = sqrt ( h );
          e[i] = scale * g;
          f = pythag ( ar[i+(i-1)*n], ai[i+(i-1)*n] );
    /*
      Form next diagonal element of matrix T.
    */
          if ( f != 0.0 )
          {
            tau[0+(i-1)*2] = ( ai[i+(i-1)*n] * tau[1+i*2]
                             - ar[i+(i-1)*n] * tau[0+i*2] ) / f;
            si =             ( ar[i+(i-1)*n] * tau[1+i*2]
                             + ai[i+(i-1)*n] * tau[0+i*2] ) / f;
            h = h + f * g;
            g = 1.0 + g / f;
            ar[i+(i-1)*n] = g * ar[i+(i-1)*n];
            ai[i+(i-1)*n] = g * ai[i+(i-1)*n];

            if ( i == 1 )
            {
              for ( j = 0; j < i; j++ )
              {
                ar[i+j*n] = scale * ar[i+j*n];
                ai[i+j*n] = scale * ai[i+j*n];
              }
              tau[1+(i-1)*2] = - si;
              hh = d[i];
              d[i] = ar[i+i*n];
              ar[i+i*n] = hh;
              ai[i+i*n] = scale * sqrt ( h );
              continue;
            }
          }
          else
          {
            tau[0+(i-1)*2] = - tau[0+i*2];
            si = tau[1+i*2];
            ar[i+(i-1)*n] = g;
          }

          f = 0.0;

          for ( j = 0; j < i; j++ )
          {
            g = 0.0;
            gi = 0.0;
    /*
      Form element of A*U.
    */
            for ( k = 0; k <= j; k++ )
            {
              g  = g  + ar[j+k*n] * ar[i+k*n] + ai[j+k*n] * ai[i+k*n];
              gi = gi - ar[j+k*n] * ai[i+k*n] + ai[j+k*n] * ar[i+k*n];
            }

            for ( k = j + 1; k < i; k++ )
            {
              g  = g  + ar[k+j*n] * ar[i+k*n] - ai[k+j*n] * ai[i+k*n];
              gi = gi - ar[k+j*n] * ai[i+k*n] - ai[k+j*n] * ar[i+k*n];
            }
    /*
      Form element of P.
    */
            e[j] = g / h;
            tau[1+j*2] = gi / h;
            f = f + e[j] * ar[i+j*n] - tau[1+j*2] * ai[i+j*n];
          }

          hh = f / ( h + h );
    /*
      Form the reduced A.
    */
          for ( j = 0; j < i; j++ )
          {
            f = ar[i+j*n];
            g = e[j] - hh * f;
            e[j] = g;
            fi = - ai[i+j*n];
            gi = tau[1+j*2] - hh * fi;
            tau[1+j*2] = - gi;

            for ( k = 0; k <= j; k++ )
            {
              ar[j+k*n] = ar[j+k*n] - f * e[k] - g * ar[i+k*n] + fi * tau[1+k*2]
                + gi * ai[i+k*n];
              ai[j+k*n] = ai[j+k*n] - f * tau[1+k*2] - g * ai[i+k*n] - fi * e[k]
                - gi * ar[i+k*n];
            }
          }

          for ( j = 0; j < i; j++ )
          {
            ar[i+j*n] = scale * ar[i+j*n];
            ai[i+j*n] = scale * ai[i+j*n];
          }
          tau[1+l*2] = - si;

          hh = d[i];
          d[i] = ar[i+i*n];
          ar[i+i*n] = hh;
          ai[i+i*n] = scale * sqrt ( h );
        }
      }

      return;
    }
    /******************************************************************************/

    int i4_max ( int i1, int i2 )

    /******************************************************************************/
    /*
      Purpose:

        I4_MAX returns the maximum of two I4's.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        29 August 2006

      Author:

        John Burkardt

      Parameters:

        Input, int I1, I2, are two integers to be compared.

        Output, int I4_MAX, the larger of I1 and I2.
    */
    {
      int value;

      if ( i2 < i1 )
      {
        value = i1;
      }
      else
      {
        value = i2;
      }
      return value;
    }
    /******************************************************************************/

    int i4_min ( int i1, int i2 )

    /******************************************************************************/
    /*
      Purpose:

        I4_MIN returns the smaller of two I4's.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        29 August 2006

      Author:

        John Burkardt

      Parameters:

        Input, int I1, I2, two integers to be compared.

        Output, int I4_MIN, the smaller of I1 and I2.
    */
    {
      int value;

      if ( i1 < i2 )
      {
        value = i1;
      }
      else
      {
        value = i2;
      }
      return value;
    }
    /******************************************************************************/

    void i4vec_print ( int n, int a[], char *title )

    /******************************************************************************/
    /*
      Purpose:

        I4VEC_PRINT prints an I4VEC.

      Discussion:

        An I4VEC is a vector of I4's.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        14 November 2003

      Author:

        John Burkardt

      Parameters:

        Input, int N, the number of components of the vector.

        Input, int A[N], the vector to be printed.

        Input, char *TITLE, a title.
    */
    {
      int i;

      fprintf ( stdout, "\n" );
      fprintf ( stdout, "%s\n", title );
      fprintf ( stdout, "\n" );

      for ( i = 0; i < n; i++ )
      {
        fprintf ( stdout, "  %6d: %8d\n", i, a[i] );
      }
      return;
    }
    /******************************************************************************/

    int imtql1 ( int n, double d[], double e[] )

    /******************************************************************************/
    /*
      Purpose:

        IMTQL1 computes all eigenvalues of a symmetric tridiagonal matrix.

      Discussion:

        This routine finds the eigenvalues of a symmetric
        tridiagonal matrix by the implicit QL method.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        03 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double D[N].  On input, the diagonal elements of
        the matrix.  On output, the eigenvalues in ascending order.  If an error
        exit is made, the eigenvalues are correct and ordered for indices
        1,2,...IERR-1, but may not be the smallest eigenvalues.

        Input/output, double E[N].  On input, the subdiagonal elements
        of the matrix in its last N-1 positions.  E(1) is arbitrary.  On output,
        E has been overwritten.

        Output, int IMTQL1, error flag.
        0, normal return,
        J, if the J-th eigenvalue has not been determined after 30 iterations.
    */
    {
      double b;
      double c;
      double f;
      double g;
      int i;
      int ierr;
      int its;
      int l;
      int m;
      double p;
      double r;
      double s;
      bool skip;
      double tst1;
      double tst2;

      ierr = 0;

      if ( n == 1 )
      {
        return ierr;
      }

      for ( i = 1; i < n; i++ )
      {
        e[i-1] = e[i];
      }
      e[n-1] = 0.0;

      for ( l = 0; l < n; l++ )
      {
        its = 0;
    /*
      Look for a small sub-diagonal element.
    */
        while ( true )
        {
          m = l;

          for ( m = l; m < n - 1; m++ )
          {
            tst1 = fabs ( d[m] ) + fabs ( d[m+1] );
            tst2 = tst1 + fabs ( e[m] );

            if ( tst2 == tst1 )
            {
              break;
            }
          }
    /*
      Order the eigenvalues.
    */
          p = d[l];

          if ( m == l )
          {
            for ( i = l; 0 <= i; i-- )
            {
              if ( i == 0 )
              {
                d[i] = p;
                break;
              }

              if ( d[i-1] <= p )
              {
                d[i] = p;
                break;
              }
             d[i] = d[i-1];
            }
            break;
          }
          else
          {
            if ( 30 <= its )
            {
              ierr = l + 1;
              return ierr;
            }
            its = its + 1;
    /*
      Form shift.
    */
            g = ( d[l+1] - p ) / ( 2.0 * e[l] );
            r = pythag ( g, 1.0 );
            g = d[m] - p + e[l] / ( g + fabs ( r ) * r8_sign ( g ) );
            s = 1.0;
            c = 1.0;
            p = 0.0;

            skip = false;

            for ( i = m - 1; l <= i; i-- )
            {
              f = s * e[i];
              b = c * e[i];
              r = pythag ( f, g );
              e[i+1] = r;
    /*
      Recover from underflow.
    */
              if ( r == 0.0 )
              {
                d[i+1] = d[i+1] - p;
                e[m] = 0.0;
                skip = true;
                break;
              }

              s = f / r;
              c = g / r;
              g = d[i+1] - p;
              r = ( d[i] - g ) * s + 2.0 * c * b;
              p = s * r;
              d[i+1] = g + p;
              g = c * r - b;
            }

            if ( ! skip )
            {
              d[l] = d[l] - p;
              e[l] = g;
              e[m] = 0.0;
            }
          }
        }
      }

      return ierr;
    }
    /******************************************************************************/

    int imtql2 ( int n, double d[], double e[], double z[] )

    /******************************************************************************/
    /*
      Purpose:

        IMTQL2 computes all eigenvalues/vectors of a symmetric tridiagonal matrix.

      Discussion:

        IMTQL2 finds the eigenvalues and eigenvectors of a symmetric tridiagonal
        matrix by the implicit QL method.

        The eigenvectors of a full symmetric matrix can also be found if TRED2
        has been used to reduce this full matrix to tridiagonal form.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        04 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double D[N].  On input, the diagonal elements of
        the input matrix.  On output, the eigenvalues in ascending order.  If an
        error exit is made, the eigenvalues are correct but
        unordered for indices 1,2,...,IERR-1.

        Input/output, double E[N].  On input, the subdiagonal elements
        of the input matrix in E(2:N).  E(1) is arbitrary.  On output, E is
        overwritten.

        Input/output, double Z[N,N].  On input, the transformation
        matrix produced in the reduction by TRED2, if performed.  If the
        eigenvectors of the tridiagonal matrix are desired, Z must contain the
        identity matrix.  On output, Z contains orthonormal eigenvectors of the
        symmetric tridiagonal (or full) matrix.  If an error exit is made, Z
        contains the eigenvectors associated with the stored eigenvalues.

        Output, int IMTQL2, error flag.
        0, for normal return,
        J, if the J-th eigenvalue has not been determined after 30 iterations.
    */
    {
      double b;
      double c;
      double f;
      double g;
      int i;
      int ierr;
      int ii;
      int its;
      int j;
      int k;
      int l;
      int m;
      double p;
      double r;
      double s;
      double t;
      double tst1;
      double tst2;

      ierr = 0;

      if ( n == 1 )
      {
        return ierr;
      }

      for ( i = 1; i < n; i++ )
      {
        e[i-1] = e[i];
      }
      e[n-1] = 0.0;

      for ( l = 0; l < n; l++ )
      {
        its = 0;
    /*
      Look for a small sub-diagonal element.
    */
        while ( true )
        {
          m = l;

          for ( m = l; m < n - 1; m++ )
          {
            tst1 = fabs ( d[m] ) + fabs ( d[m+1] );
            tst2 = tst1 + fabs ( e[m] );

            if ( tst2 == tst1 )
            {
              break;
            }
          }

          p = d[l];

          if ( m == l )
          {
            break;
          }

          if ( 30 <= its )
          {
            ierr = l + 1;
            return ierr;
          }

          its = its + 1;
    /*
      Form shift.
    */
          g = ( d[l+1] - p ) / ( 2.0 * e[l] );
          r = pythag ( g, 1.0 );
          g = d[m] - p + e[l] / ( g + fabs ( r ) * r8_sign ( g ) );
          s = 1.0;
          c = 1.0;
          p = 0.0;

          for ( i = m - 1; l <= i; i-- )
          {
            f = s * e[i];
            b = c * e[i];
            r = pythag ( f, g );
            e[i+1] = r;
    /*
      Recover from underflow.
    */
            if ( r == 0.0 )
            {
              d[i+1] = d[i+1] - p;
              e[m] = 0.0;
              continue;
            }

            s = f / r;
            c = g / r;
            g = d[i+1] - p;
            r = ( d[i] - g ) * s + 2.0 * c * b;
            p = s * r;
            d[i+1] = g + p;
            g = c * r - b;
    /*
      Form vector.
    */
            for ( k = 0; k < n; k++ )
            {
              f = z[k+(i+1)*n];
              z[k+(i+1)*n] = s * z[k+i*n] + c * f;
              z[k+i*n]     = c * z[k+i*n] - s * f;
            }
          }

          d[l] = d[l] - p;
          e[l] = g;
          e[m] = 0.0;
        }
      }
    /*
      Order eigenvalues and eigenvectors.
    */
      for ( i = 0; i < n - 1; i++ )
      {
        k = i;
        p = d[i];

        for ( j = i + 1; j < n; j++ )
        {
          if ( d[j] < p )
          {
            k = j;
            p = d[j];
          }
        }

        if ( k != i )
        {
          d[k] = d[i];
          d[i] = p;

          for ( ii = 0; ii < n; ii++ )
          {
            t         = z[ii+i*n];
            z[ii+i*n] = z[ii+k*n];
            z[ii+k*n] = t;
          }
        }
      }

      return ierr;
    }
    /******************************************************************************/

    int imtqlv ( int n, double d[], double e[], double e2[], double w[], int ind[] )

    /******************************************************************************/
    /*
      Purpose:

        IMTQLV computes all eigenvalues of a real symmetric tridiagonal matrix.

      Discussion:

        IMTQLV finds the eigenvalues of a symmetric tridiagonal matrix by
        the implicit QL method and associates with them their corresponding
        submatrix indices.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        13 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double D[N], the diagonal elements of the input matrix.

        Input, double E[N], the subdiagonal elements of the input matrix
        in E(2:N).  E(1) is arbitrary.

        Input/output, double E2[N].  On input, the squares of the
        corresponding elements of E.  E2(1) is arbitrary.  On output, elements of
        E2 corresponding to elements of E regarded as negligible have been
        replaced by zero, causing the matrix to split into a direct sum of
        submatrices.  E2(1) is also set to zero.

        Output, double W[N], the eigenvalues in ascending order.  If an
        error break; is made, the eigenvalues are correct and ordered for
        indices 1,2,...IERR-1, but may not be the smallest eigenvalues.

        Output, int IND[N], the submatrix indices associated with
        the corresponding eigenvalues in W: 1 for eigenvalues belonging to the
        first submatrix from the top, 2 for those belonging to the second
        submatrix, and so on.

        Output, int IMTQLV, error flag.
        0, for normal return,
        J, if the J-th eigenvalue has not been determined after 30 iterations.
    */
    {
      double b;
      double c;
      double f;
      double g;
      int i;
      int ierr;
      int its;
      int k;
      int l;
      int m;
      double p;
      double r;
      double *rv1;
      double s;
      bool skip;
      int tag;
      double tst1;
      double tst2;

      ierr = 0;

      k = -1;
      tag = -1;
      for ( i = 0; i < n; i++ )
      {
        w[i] = d[i];
      }
      e2[0] = 0.0;

      rv1 = ( double * ) malloc ( n * sizeof ( double ) );
      for ( i = 0; i < n - 1; i++ )
      {
        rv1[i] = e[i+1];
      }
      rv1[n-1] = 0.0;

      for ( l = 0; l < n; l++ )
      {
        its = 0;
    /*
      Look for a small sub-diagonal element.
    */
        while ( true )
        {
          for ( m = l; m < n; m++ )
          {
            if ( m == n - 1 )
            {
              break;
            }

            tst1 = fabs ( w[m] ) + fabs ( w[m+1] );
            tst2 = tst1 + fabs ( rv1[m] );

            if ( tst2 == tst1 )
            {
              break;
            }
    /*
      Guard against underflowed element of E2.
    */
            if ( e2[m+1] == 0.0 )
            {
              k = m;
              tag = tag + 1;
              break;
            }
          }

          if ( k < m )
          {
            if ( m < n - 1 )
            {
              e2[m+1] = 0.0;
            }
            k = m;
            tag = tag + 1;
          }

          p = w[l];

          if ( m == l )
          {
            for ( i = l; 0 <= i; i-- )
            {
              if ( i == 0 )
              {
                w[i] = p;
                ind[i] = tag;
              }
              else if ( w[i-1] <= p )
              {
                w[i] = p;
                ind[i] = tag;
                break;
              }
              else
              {
                w[i] = w[i-1];
                ind[i] = ind[i-1];
              }
            }

            break;
          }
          else
          {
            if ( 30 <= its )
            {
              free ( rv1 );
              ierr = l + 1;
              return ierr;
            }

            its = its + 1;
    /*
      Form shift.
    */
            g = ( w[l+1] - p ) / ( 2.0 * rv1[l] );
            r = pythag ( g, 1.0 );
            g = w[m] - p + rv1[l] / ( g + fabs ( r ) * r8_sign ( g ) );
            s = 1.0;
            c = 1.0;
            p = 0.0;

            skip = false;

            for ( i = m - 1; l <= i; i-- )
            {
              f = s * rv1[i];
              b = c * rv1[i];
              r = pythag ( f, g );
              rv1[i+1] = r;

              if ( r == 0.0 )
              {
                w[i+1] = w[i+1] - p;
                rv1[m] = 0.0;
                skip = true;
                break;
              }

              s = f / r;
              c = g / r;
              g = w[i+1] - p;
              r = ( w[i] - g ) * s + 2.0 * c * b;
              p = s * r;
              w[i+1] = g + p;
              g = c * r - b;
            }

            if ( ! skip )
            {
              w[l] = w[l] - p;
              rv1[l] = g;
              rv1[m] = 0.0;
            }
          }
        }
      }

      free ( rv1 );

      return ierr;
    }
    /******************************************************************************/


    /******************************************************************************/


    /******************************************************************************/

    void ortbak ( int n, int low, int igh, double a[], double ort[], int m,
      double z[] )

    /******************************************************************************/
    /*
      Purpose:

        ORTBAK determines eigenvectors by undoing the ORTHES transformation.

      Discussion:

        ORTBAK forms the eigenvectors of a real general
        matrix by back transforming those of the corresponding
        upper Hessenberg matrix determined by ORTHES.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        01 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine BALANC.  If BALANC has not been used, set LOW = 1 and IGH equal
        to the order of the matrix.

        Input, double A[N,IGH], contains information about the
        orthogonal transformations used in the reduction by ORTHES in its strict
        lower triangle.

        Input/output, double ORT[IGH], contains further information
        about the transformations used in the reduction by ORTHES.  On output, ORT
        has been altered.

        Input, int M, the number of columns of Z to be back
        transformed.

        Input/output, double Z[N,N].  On input, the real and imaginary
        parts of the eigenvectors to be back transformed in the first M columns.
        On output, the real and imaginary parts of the transformed eigenvectors.
    */
    {
      double g;
      int i;
      int j;
      int mp;

      if ( m == 0 )
      {
        return;
      }

      for ( mp = igh - 1; low + 1 <= mp; mp-- )
      {
        if ( a[mp+(mp-1)*n] != 0.0 )
        {
          for ( i = mp + 1; i <= igh; i++ )
          {
            ort[i] = a[i+(mp-1)*n];
          }
          for ( j = 0; j < m; j++ )
          {
            g = 0.0;
            for ( i = mp; i <= igh; i++ )
            {
              g = g + ort[i] * z[i+j*n];
            }
            g = ( g / ort[mp] ) / a[mp+(mp-1)*n];
            for ( i = mp; i <= igh; i++ )
            {
              z[i+j*n] = z[i+j*n] + g * ort[i];
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void orthes ( int n, int low, int igh, double a[], double ort[] )

    /******************************************************************************/
    /*
      Purpose:

        ORTHES transforms a real general matrix to upper Hessenberg form.

      Discussion:

        ORTHES is given a real general matrix, and reduces a submatrix
        situated in rows and columns LOW through IGH to upper Hessenberg form by
        orthogonal similarity transformations.

      Modified:

        03 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine BALANC.  If BALANC has not been used, set LOW = 1 and IGH = N.

        Input/output, double A[N,N].  On input, the matrix.  On output,
        the Hessenberg matrix.  Information about the orthogonal transformations
        used in the reduction is stored in the remaining triangle under the
        Hessenberg matrix.

        Output, double ORT[IGH], contains further information about the
        transformations.
    */
    {
      double f;
      double g;
      double h;
      int i;
      //int ii;
      int j;
      //int jj;
      //int la;
      int m;
      double scale;

      for ( m = low + 1; m <= igh - 1; m++ )
      {
        h = 0.0;
        ort[m] = 0.0;
        scale = 0.0;
    /*
      Scale the column.
    */
        for ( i = m; i <= igh; i++ )
        {
          scale = scale + fabs ( a[i+(m-1)*n] );
        }

        if ( scale != 0.0 )
        {
          for ( i = igh; m <= i; i-- )
          {
            ort[i] = a[i+(m-1)*n] / scale;
            h = h + ort[i] * ort[i];
          }

          g = - sqrt ( h ) * r8_sign ( ort[m] );
          h = h - ort[m] * g;
          ort[m] = ort[m] - g;
    /*
      Form (I-(U*Ut)/h) * A.
    */
          for ( j = m; j < n; j++ )
          {
            f = 0.0;
            for ( i = igh; m <= i; i-- )
            {
              f = f + ort[i] * a[i+j*n];
            }
            f = f / h;

            for ( i = m; i <= igh; i++ )
            {
              a[i+j*n] = a[i+j*n] - f * ort[i];
            }
          }
    /*
      Form (I-(u*ut)/h) * A * (I-(u*ut)/h).
    */
          for ( i = 0; i <= igh; i++ )
          {
            f = 0.0;
            for ( j = igh; m <= j; j-- )
            {
              f = f + ort[j] * a[i+j*n];
            }
            for ( j = m; j <= igh; j++ )
            {
              a[i+j*n] = a[i+j*n] - f * ort[j] / h;
            }
          }
          ort[m] = scale * ort[m];
          a[m+(m-1)*n] = scale * g;
        }
      }

      return;
    }
    /******************************************************************************/

    void ortran ( int n, int low, int igh, double a[], double ort[], double z[] )

    /******************************************************************************/
    /*
      Purpose:

        ORTRAN accumulates similarity transformations generated by ORTHES.

      Discussion:

        ORTRAN accumulates the orthogonal similarity
        transformations used in the reduction of a real general
        matrix to upper Hessenberg form by ORTHES.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        01 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int LOW, IGH, are determined by the balancing
        routine BALANC.  If BALANC has not been used, set LOW = 1, IGH = N.

        Input, double A[N,IGH], contains information about the
        orthogonal transformations used in the reduction by ORTHES in its strict
        lower triangle.

        Input/output, double ORT[IGH], contains further information
        about the transformations used in the reduction by ORTHES.  On output, ORT
        has been further altered.

        Output, double Z[N,N], contains the transformation matrix
        produced in the reduction by ORTHES.
    */
    {
      double g;
      int i;
      int j;
      int mp;
    /*
      Initialize Z to the identity matrix.
    */
      r8mat_identity ( n, z );

      if ( igh - low < 2 )
      {
        return;
      }

      for ( mp = igh - 1; low + 1 <= mp; mp-- )
      {
        if ( a[mp+(mp-1)*n] != 0.0 )
        {
          for ( i = mp + 1; i <= igh; i++ )
          {
            ort[i] = a[i+(mp-1)*n];
          }
          for ( j = mp; j <= igh; j++ )
          {
            g = 0.0;
            for ( i = mp; i <= igh; i++ )
            {
              g = g + ort[i] * z[i+j*n];
            }
            g = ( g / ort[mp] ) / a[mp+(mp-1)*n];

            for ( i = mp; i <= igh; i++ )
            {
              z[i+j*n] = z[i+j*n] + g * ort[i];
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    double pythag ( double a, double b )

    /******************************************************************************/
    /*
      Purpose:

        PYTHAG computes SQRT ( A * A + B * B ) carefully.

      Discussion:

        The formula

          PYTHAG = sqrt ( A * A + B * B )

        is reasonably accurate, but can fail if, for example, A^2 is larger
        than the machine overflow.  The formula can lose most of its accuracy
        if the sum of the squares is very large or very small.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Modified:

        08 November 2012

      Parameters:

        Input, double A, B, the two legs of a right triangle.

        Output, double PYTHAG, the length of the hypotenuse.
    */
    {
      double p;
      double r;
      double s;
      double t;
      double u;

      p = r8_max ( fabs ( a ), fabs ( b ) );

      if ( p != 0.0 )
      {
        r = r8_min ( fabs ( a ), fabs ( b ) ) / p;
        r = r * r;

        while ( 1 )
        {
          t = 4.0 + r;

          if ( t == 4.0 )
          {
            break;
          }

          s = r / t;
          u = 1.0 + 2.0 * s;
          p = u * p;
          r = ( s / u ) * ( s / u ) * r;
        }
      }
      return p;
    }
    /******************************************************************************/

    void qzhes ( int n, double a[], double b[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        QZHES carries out transformations for a generalized eigenvalue problem.

      Discussion:

        QZHES is the first step of the QZ algorithm
        for solving generalized matrix eigenvalue problems.

        QZHES accepts a pair of real general matrices and
        reduces one of them to upper Hessenberg form and the other
        to upper triangular form using orthogonal transformations.
        it is usually followed by QZIT, QZVAL and, possibly, QZVEC.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        10 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrices.

        Input/output, double A[N,N].  On input, the first real general
        matrix.  On output, A has been reduced to upper Hessenberg form.  The
        elements below the first subdiagonal have been set to zero.

        Input/output, double B[N,N].  On input, a real general matrix.
        On output, B has been reduced to upper triangular form.  The elements
        below the main diagonal have been set to zero.

        Input, bool MATZ, is true if the right hand transformations
        are to be accumulated for later use in computing eigenvectors.

        Output, double Z[N,N], contains the product of the right hand
        transformations if MATZ is true.
    */
    {
      int i;
      int j;
      int k;
      int l;
      double r;
      double rho;
      double s;
      double t;
      double u1;
      double u2;
      double v1;
      double v2;
    /*
      Set Z to the identity matrix.
    */
      if ( matz )
      {
        r8mat_identity ( n, z );
      }
    /*
      Reduce B to upper triangular form.
    */
      if ( n <= 1 )
      {
        return;
      }

      for ( l = 0; l < n - 1; l++ )
      {
        s = 0.0;
        for ( i = l + 1; i < n; i++ )
        {
          s = s + fabs ( b[i+l*n] );
        }

        if ( s != 0.0 )
        {
          s = s + fabs ( b[l+l*n] );
          for ( i = l; i < n; i++ )
          {
            b[i+l*n] = b[i+l*n] / s;
          }
          r = 0.0;
          for ( i = l; i < n; i++ )
          {
            r = r + b[i+l*n] * b[i+l*n];
          }
          r = sqrt ( r );
          r = fabs ( r ) * r8_sign ( b[l+l*n] );
          b[l+l*n] = b[l+l*n] + r;
          rho = r * b[l+l*n];

          for ( j = l + 1; j < n; j++ )
          {
            t = 0.0;
            for ( i = l; i < n; i++ )
            {
              t = t + b[i+l*n] * b[i+j*n];
            }
            for ( i = l; i < n; i++ )
            {
              b[i+j*n] = b[i+j*n] - t * b[i+l*n] / rho;
            }
          }

          for ( j = 0; j < n; j++ )
          {
            t = 0.0;
            for ( i = l; i < n; i++ )
            {
              t = t + b[i+l*n] * a[i+j*n];
            }
            for ( i = l; i < n; i++ )
            {
              a[i+j*n] = a[i+j*n] - t * b[i+l*n] / rho;
            }
          }

          b[l+l*n] = - s * r;
          for ( i = l + 1; i < n; i++ )
          {
            b[i+l*n] = 0.0;
          }
        }
      }
    /*
      Reduce A to upper Hessenberg form, while keeping B triangular.
    */
      for ( k = 0; k < n - 2; k++ )
      {
        for ( l = n - 2; k + 1 <= l; l-- )
        {
    /*
      Zero A[l+1+k*n].
    */
          s = fabs ( a[l+k*n] ) + fabs ( a[l+1+k*n] );

          if ( s != 0.0 )
          {
            u1 = a[l+k*n] / s;
            u2 = a[l+1+k*n] / s;
            r = sqrt ( u1 * u1 + u2 * u2 ) * r8_sign ( u1 );
            v1 = - ( u1 + r ) / r;
            v2 = - u2 / r;
            u2 = v2 / v1;

            for ( j = k; j < n; j++ )
            {
              t = a[l+j*n] + u2 * a[l+1+j*n];
              a[l+j*n] = a[l+j*n] + t * v1;
              a[l+1+j*n] = a[l+1+j*n] + t * v2;
            }

            a[l+1+k*n] = 0.0;

            for ( j = l; j < n; j++ )
            {
              t = b[l+j*n] + u2 * b[l+1+j*n];
              b[l+j*n] = b[l+j*n] + t * v1;
              b[l+1+j*n] = b[l+1+j*n] + t * v2;
            }
    /*
      Zero B[l+1+l*n].
    */
            s = fabs ( b[l+1+(l+1)*n] ) + fabs ( b[l+1+l*n] );

            if ( s != 0.0 )
            {
              u1 = b[l+1+(l+1)*n] / s;
              u2 = b[l+1+l*n] / s;
              r = sqrt ( u1 * u1 + u2 * u2 ) * r8_sign ( u1 );
              v1 =  - ( u1 + r ) / r;
              v2 = - u2 / r;
              u2 = v2 / v1;

              for ( i = 0; i <= l + 1; i++ )
              {
                t = b[i+(l+1)*n] + u2 * b[i+l*n];
                b[i+(l+1)*n] = b[i+(l+1)*n] + t * v1;
                b[i+l*n] = b[i+l*n] + t * v2;
              }

              b[l+1+l*n] = 0.0;

              for ( i = 0; i < n; i++ )
              {
                t = a[i+(l+1)*n] + u2 * a[i+l*n];
                a[i+(l+1)*n] = a[i+(l+1)*n] + t * v1;
                a[i+l*n] = a[i+l*n] + t * v2;
              }

              if ( matz )
              {
                for ( i = 0; i < n; i++ )
                {
                  t = z[i+(l+1)*n] + u2 * z[i+l*n];
                  z[i+(l+1)*n] = z[i+(l+1)*n] + t * v1;
                  z[i+l*n] = z[i+l*n] + t * v2;
                }
              }
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/


    double r8_epsilon ( )

    /******************************************************************************/
    /*
      Purpose:

        R8_EPSILON returns the R8 round off unit.

      Discussion:

        R8_EPSILON is a number R which is a power of 2 with the property that,
        to the precision of the computer's arithmetic,
          1 < 1 + R
        but
          1 = ( 1 + R / 2 )

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        01 September 2012

      Author:

        John Burkardt

      Parameters:

        Output, double R8_EPSILON, the R8 round-off unit.
    */
    {
      const double value = 2.220446049250313E-016;

      return value;
    }
    /******************************************************************************/

    double r8_max ( double x, double y )

    /******************************************************************************/
    /*
      Purpose:

        R8_MAX returns the maximum of two R8's.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        07 May 2006

      Author:

        John Burkardt

      Parameters:

        Input, double X, Y, the quantities to compare.

        Output, double R8_MAX, the maximum of X and Y.
    */
    {
      double value;

      if ( y < x )
      {
        value = x;
      }
      else
      {
        value = y;
      }
      return value;
    }
    /******************************************************************************/

    double r8_min ( double x, double y )

    /******************************************************************************/
    /*
      Purpose:

        R8_MIN returns the minimum of two R8's.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        07 May 2006

      Author:

        John Burkardt

      Parameters:

        Input, double X, Y, the quantities to compare.

        Output, double R8_MIN, the minimum of X and Y.
    */
    {
      double value;

      if ( y < x )
      {
        value = y;
      }
      else
      {
        value = x;
      }
      return value;
    }
    /******************************************************************************/

    double r8_sign ( double x )

    /******************************************************************************/
    /*
      Purpose:

        R8_SIGN returns the sign of an R8.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 May 2006

      Author:

        John Burkardt

      Parameters:

        Input, double X, the number whose sign is desired.

        Output, double R8_SIGN, the sign of X.
    */
    {
      double value;

      if ( x < 0.0 )
      {
        value = - 1.0;
      }
      else
      {
        value = + 1.0;
      }
      return value;
    }
    /******************************************************************************/

    void r8mat_identity  ( int n, double a[] )

    /******************************************************************************/
    /*
      Purpose:

        R8MAT_IDENTITY sets an R8MAT to the identity matrix.

      Discussion:

        An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
        in column-major order.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        06 September 2005

      Author:

        John Burkardt

      Parameters:

        Input, int N, the order of A.

        Output, double A[N*N], the N by N identity matrix.
    */
    {
      int i;
      int j;
      int k;

      k = 0;
      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < n; i++ )
        {
          if ( i == j )
          {
            a[k] = 1.0;
          }
          else
          {
            a[k] = 0.0;
          }
          k = k + 1;
        }
      }

      return;
    }
    /******************************************************************************/

    double *r8mat_mm_new ( int n1, int n2, int n3, double a[], double b[] )

    /******************************************************************************/
    /*
      Purpose:

        R8MAT_MM_NEW multiplies two matrices.

      Discussion:

        An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
        in column-major order.

        For this routine, the result is returned as the function value.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 April 2009

      Author:

        John Burkardt

      Parameters:

        Input, int N1, N2, N3, the order of the matrices.

        Input, double A[N1*N2], double B[N2*N3], the matrices to multiply.

        Output, double R8MAT_MM[N1*N3], the product matrix C = A * B.
    */
    {
      double *c;
      int i;
      int j;
      int k;

      c = ( double * ) malloc ( n1 * n3 * sizeof ( double ) );

      for ( i = 0; i < n1; i ++ )
      {
        for ( j = 0; j < n3; j++ )
        {
          c[i+j*n1] = 0.0;
          for ( k = 0; k < n2; k++ )
          {
            c[i+j*n1] = c[i+j*n1] + a[i+k*n1] * b[k+j*n2];
          }
        }
      }

      return c;
    }
    /******************************************************************************/

    double *r8mat_mmt_new ( int n1, int n2, int n3, double a[], double b[] )

    /******************************************************************************/
    /*
      Purpose:

        R8MAT_MMT_NEW computes C = A * B'.

      Discussion:

        An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
        in column-major order.

        For this routine, the result is returned as the function value.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        13 November 2012

      Author:

        John Burkardt

      Parameters:

        Input, int N1, N2, N3, the order of the matrices.

        Input, double A[N1*N2], double B[N3*N2], the matrices to multiply.

        Output, double R8MAT_MMT_NEW[N1*N3], the product matrix C = A * B'.
    */
    {
      double *c;
      int i;
      int j;
      int k;

      c = ( double * ) malloc ( n1 * n3 * sizeof ( double ) );

      for ( i = 0; i < n1; i++ )
      {
        for ( j = 0; j < n3; j++ )
        {
          c[i+j*n1] = 0.0;
          for ( k = 0; k < n2; k++ )
          {
            c[i+j*n1] = c[i+j*n1] + a[i+k*n1] * b[j+k*n3];
          }
        }
      }

      return c;
    }
    /******************************************************************************/

    void r8mat_print ( int m, int n, double a[], char *title )

    /******************************************************************************/
    /*
      Purpose:

        R8MAT_PRINT prints an R8MAT.

      Discussion:

        An R8MAT is a doubly dimensioned array of R8's, which
        may be stored as a vector in column-major order.

        Entry A(I,J) is stored as A[I+J*M]

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        28 May 2008

      Author:

        John Burkardt

      Parameters:

        Input, int M, the number of rows in A.

        Input, int N, the number of columns in A.

        Input, double A[M*N], the M by N matrix.

        Input, char *TITLE, a title.
    */
    {
      r8mat_print_some ( m, n, a, 1, 1, m, n, title );

      return;
    }
    /******************************************************************************/

    void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi,
      int jhi, char *title )

    /******************************************************************************/
    /*
      Purpose:

        R8MAT_PRINT_SOME prints some of an R8MAT.

      Discussion:

        An R8MAT is a doubly dimensioned array of R8's, which
        may be stored as a vector in column-major order.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        20 August 2010

      Author:

        John Burkardt

      Parameters:

        Input, int M, the number of rows of the matrix.
        M must be positive.

        Input, int N, the number of columns of the matrix.
        N must be positive.

        Input, double A[M*N], the matrix.

        Input, int ILO, JLO, IHI, JHI, designate the first row and
        column, and the last row and column to be printed.

        Input, char *TITLE, a title.
    */
    {
    # define INCX 5

      int i;
      int i2hi;
      int i2lo;
      int j;
      int j2hi;
      int j2lo;

      fprintf ( stdout, "\n" );
      fprintf ( stdout, "%s\n", title );

      if ( m <= 0 || n <= 0 )
      {
        fprintf ( stdout, "\n" );
        fprintf ( stdout, "  (None)\n" );
        return;
      }
    /*
      Print the columns of the matrix, in strips of 5.
    */
      for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX )
      {
        j2hi = j2lo + INCX - 1;
        j2hi = i4_min ( j2hi, n );
        j2hi = i4_min ( j2hi, jhi );

        fprintf ( stdout, "\n" );
    /*
      For each column J in the current range...

      Write the header.
    */
        fprintf ( stdout, "  Col:  ");
        for ( j = j2lo; j <= j2hi; j++ )
        {
          fprintf ( stdout, "  %7d     ", j - 1 );
        }
        fprintf ( stdout, "\n" );
        fprintf ( stdout, "  Row\n" );
        fprintf ( stdout, "\n" );
    /*
      Determine the range of the rows in this strip.
    */
        i2lo = i4_max ( ilo, 1 );
        i2hi = i4_min ( ihi, m );

        for ( i = i2lo; i <= i2hi; i++ )
        {
    /*
      Print out (up to) 5 entries in row I, that lie in the current strip.
    */
          fprintf ( stdout, "%5d:", i - 1 );
          for ( j = j2lo; j <= j2hi; j++ )
          {
            fprintf ( stdout, "  %14f", a[i-1+(j-1)*m] );
          }
          fprintf ( stdout, "\n" );
        }
      }

      return;
    # undef INCX
    }
    /******************************************************************************/

    double *r8mat_uniform_01_new ( int m, int n, int *seed )

    /******************************************************************************/
    /*
      Purpose:

        R8MAT_UNIFORM_01_NEW fills an R8MAT with pseudorandom values scaled to [0,1].

      Discussion:

        An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
        in column-major order.

        This routine implements the recursion

          seed = 16807 * seed mod ( 2^31 - 1 )
          unif = seed / ( 2^31 - 1 )

        The integer arithmetic never requires more than 32 bits,
        including a sign bit.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        30 June 2009

      Author:

        John Burkardt

      Reference:

        Paul Bratley, Bennett Fox, Linus Schrage,
        A Guide to Simulation,
        Springer Verlag, pages 201-202, 1983.

        Bennett Fox,
        Algorithm 647:
        Implementation and Relative Efficiency of Quasirandom
        Sequence Generators,
        ACM Transactions on Mathematical Software,
        Volume 12, Number 4, pages 362-376, 1986.

        Philip Lewis, Allen Goodman, James Miller,
        A Pseudo-Random Number Generator for the System/360,
        IBM Systems Journal,
        Volume 8, pages 136-143, 1969.

      Parameters:

        Input, int M, N, the number of rows and columns.

        Input/output, int *SEED, the "seed" value.  Normally, this
        value should not be 0, otherwise the output value of SEED
        will still be 0, and R8_UNIFORM will be 0.  On output, SEED has
        been updated.

        Output, double R8MAT_UNIFORM_01_NEW[M*N], a matrix of pseudorandom values.
    */
    {
      int i;
      int j;
      int k;
      double *r;

      r = ( double * ) malloc ( m * n * sizeof ( double ) );

      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          k = *seed / 127773;

          *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

          if ( *seed < 0 )
          {
            *seed = *seed + 2147483647;
          }
          r[i+j*m] = ( double ) ( *seed ) * 4.656612875E-10;
        }
      }

      return r;
    }
    /******************************************************************************/

    void r8vec_print ( int n, double a[], char *title )

    /******************************************************************************/
    /*
      Purpose:

        R8VEC_PRINT prints an R8VEC.

      Discussion:

        An R8VEC is a vector of R8's.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 April 2009

      Author:

        John Burkardt

      Parameters:

        Input, int N, the number of components of the vector.

        Input, double A[N], the vector to be printed.

        Input, char *TITLE, a title.
    */
    {
      int i;

      fprintf ( stdout, "\n" );
      fprintf ( stdout, "%s\n", title );
      fprintf ( stdout, "\n" );
      for ( i = 0; i < n; i++ )
      {
        fprintf ( stdout, "  %8d: %14f\n", i, a[i] );
      }

      return;
    }
    /******************************************************************************/

    int ratqr ( int n, double eps1, double d[], double e[], double e2[], int m,
      double w[], int ind[], double bd[], bool type, int idef )

    /******************************************************************************/
    /*
      Purpose:

        RATQR computes selected eigenvalues of a real symmetric tridiagonal matrix.

      Discussion:

        RATQR finds the algebraically smallest or largest eigenvalues of a
        symmetric tridiagonal matrix by the rational QR method with Newton
        corrections.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        03 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double EPS1.  On input, a theoretical absolute
        error tolerance for the computed eigenvalues.  If the input EPS1 is
        non-positive, or indeed smaller than its default value, it is reset at
        each iteration to the respective default value, namely, the product of
        the relative machine precision and the magnitude of the current eigenvalue
        iterate.  The theoretical absolute error in the K-th eigenvalue is usually
        not greater than K times EPS1.  On output, EPS1 is unaltered unless it has
        been reset to its (last) default value.

        Input, double D[N], the diagonal elements of the input matrix.

        Input, double E[N], the subdiagonal elements of the input matrix
        in E(2:N).  E(1) is arbitrary.

        Input/output, double E2[N].  On input, E2(2:N-1) contains the
        squares of the corresponding elements of E, and E2(1) is arbitrary.  On
        output, elements of E2 corresponding to elements of E regarded as
        negligible have been replaced by zero, causing the matrix to split into
        a direct sum of submatrices.  E2(1) is set to 0.0D+00 if the smallest
        eigenvalues have been found, and to 2.0D+00 if the largest eigenvalues
        have been found.  E2 is otherwise unaltered (unless overwritten by BD).

        Input, int M, the number of eigenvalues to be found.

        Output, double W[M], the M algebraically smallest eigenvalues in
        ascending order, or the M largest eigenvalues in descending order.
        If an error exit is made because of an incorrect specification of IDEF,
        no eigenvalues are found.  If the Newton iterates for a particular
        eigenvalue are not monotone, the best estimate obtained is returned
        and IERR is set.  W may coincide with D.

        Output, int IND[N], contains in its first M positions the submatrix
        indices associated with the corresponding eigenvalues in W:
        1 for eigenvalues belonging to the first submatrix from the top, 2 for
        those belonging to the second submatrix, and so on.

        Output, double BD[N], contains refined bounds for the
        theoretical errors of the corresponding eigenvalues in W.  These bounds
        are usually within the tolerance specified by EPS1.  BD may coincide
        with E2.

        Input, bool TYPE, should be set to TRUE if the smallest eigenvalues
        are to be found, and to FALSE if the largest eigenvalues are to be found.

        Input, int IDEF, should be set to 1 if the input matrix
        is known to be positive definite, to -1 if the input matrix is known to
        be negative  definite, and to 0 otherwise.

        Output, int RATQR, error flag.
        0, for normal return,
        6*N+1, if IDEF is set to 1 and TYPE to .true. when the matrix is not
          positive definite, or if IDEF is set to -1 and TYPE to .false.
          when the matrix is not negative definite,
        5*N+K, if successive iterates to the K-th eigenvalue are not monotone
          increasing, where K refers to the last such occurrence.
    */
    {
      double delta;
      double ep;
      double err;
      double f;
      int i;
      int ierr;
      int ii;
      bool irreg;
      int j;
      int jdef;
      int k;
      double p;
      double q;
      double qp;
      double r;
      double s;
      double tot;

      ierr = 0;
      jdef = idef;
      for ( i = 0; i < n; i++ )
      {
        w[i] = d[i];
      }

      if ( ! type )
      {
        for ( i = 0; i < n; i++ )
        {
          w[i] = - w[i];
        }
        jdef = - jdef;
      }

      while ( true )
      {
        err = 0.0;
        s = 0.0;
    /*
      Look for small sub-diagonal entries and define initial shift
      from lower Gerschgorin bound.

      Copy E2 array into BD.
    */
        tot = w[0];
        q = 0.0;
        j = -1;

        for ( i = 0; i < n; i++ )
        {
          p = q;

          if ( i == 0 )
          {
            e2[i] = 0.0;
          }
          else if ( p <= ( fabs ( d[i] ) + fabs (  d[i-1] ) ) * r8_epsilon ( ) )
          {
            e2[i] = 0.0;
          }

          bd[i] = e2[i];
    /*
      Count also if element of E2 has underflowed.
    */
          if ( e2[i] == 0.0 )
          {
            j = j + 1;
          }

          ind[i] = j;
          q = 0.0;
          if ( i < n - 1 )
          {
            q = fabs ( e[i+1] );
          }

          tot = r8_min ( tot, w[i] - p - q );
        }

        if ( jdef == 1 && tot < 0.0 )
        {
          tot = 0.0;
        }
        else
        {
          for ( i = 0; i < n; i++ )
          {
            w[i] = w[i] - tot;
          }
        }

        for ( k = 0; k < m; k++ )
        {
    /*
      Next QR transformation.
    */
          irreg = true;

          while ( true )
          {
            tot = tot + s;
            delta = w[n-1] - s;
            i = n - 1;
            f = fabs ( tot ) * r8_epsilon ( );
            eps1 = r8_max ( eps1, f );

            if ( delta <= eps1 )
            {
              if ( delta < - eps1 )
              {
                ierr = 6 * n + 1;
                return ierr;
              }

              irreg = false;
              break;
            }
    /*
      Replace small sub-diagonal squares by zero to reduce the incidence of
      underflows.
    */
            for ( j = k + 1; j < n; j++ )
            {
              if ( bd[j] <= r8_epsilon ( ) * r8_epsilon ( ) )
              {
                bd[j] = 0.0;
              }
            }

            f = bd[n-1] / delta;
            qp = delta + f;
            p = 1.0;

            for ( i = n - 2; k <= i; i-- )
            {
              q = w[i] - s - f;
              r = q / qp;
              p = p * r + 1.0;
              ep = f * r;
              w[i+1] = qp + ep;
              delta = q - ep;

              if ( delta <= eps1 )
              {
                if ( delta < - eps1 )
                {
                  ierr = 6 * n + 1;
                  return ierr;
                }
                irreg = false;
                break;
              }

              f = bd[i] / q;
              qp = delta + f;
              bd[i+1] = qp * ep;
            }

            if ( ! irreg )
            {
              break;
            }

            w[k] = qp;
            s = qp / p;

            if ( tot + s <= tot )
            {
              break;
            }
          }
    /*
      Set error: irregular end of iteration.
      Deflate minimum diagonal element.
    */
          if ( irreg )
          {
            ierr = 5 * n + k;
            s = 0.0;
            delta = qp;

            for ( j = k; j < n; j++ )
            {
              if ( w[j] <= delta )
              {
                i = j;
                delta = w[j];
              }
            }
          }
    /*
      Convergence.
    */
          if ( i < n - 1 )
          {
            bd[i+1] = bd[i] * f / qp;
          }

          ii = ind[i];

          for ( j = i - 1; k <= j; j-- )
          {
            w[j+1] = w[j] - s;
            bd[j+1] = bd[j];
            ind[j+1] = ind[j];
          }

          w[k] = tot;
          err = err + fabs ( delta );
          bd[k] = err;
          ind[k] = ii;
        }

        if ( type )
        {
          return ierr;
        }

        f = bd[0];
        e2[0] = 2.0;
        bd[0] = f;
    /*
      Negate elements of W for largest values.
    */
        for ( i = 0; i < n; i++ )
        {
          w[i] = - w[i];
        }
        jdef = - jdef;

        break;
      }

      return ierr;
    }
    /******************************************************************************/

    void rebak ( int n, double b[], double dl[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        REBAK determines eigenvectors by undoing the REDUC transformation.

      Discussion:

        REBAK forms the eigenvectors of a generalized
        symmetric eigensystem by back transforming those of the
        derived symmetric matrix determined by REDUC.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        24 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        FORTRAN90 version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double B[N,N], contains information about the similarity
        transformation (Cholesky decomposition) used in the reduction by REDUC
        in its strict lower triangle.

        Input, double DL[N], further information about the
        transformation.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double Z[N,M].  On input, the eigenvectors to be
        back transformed in its first M columns.  On output, the transformed
        eigenvectors.
    */
    {
      double dot;
      int i;
      int j;
      int k;

      for ( j = 0; j < m; j++ )
      {
        for ( i = n - 1; 0 <= i; i-- )
        {
          dot = 0.0;
          for ( k = i + 1; k < n; k++ )
          {
            dot = dot + b[k+i*n] * z[k+j*n];
          }
          z[i+j*n] = ( z[i+j*n] - dot ) / dl[i];
        }
      }

      return;
    }
    /******************************************************************************/

    void rebakb ( int n, double b[], double dl[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        REBAKB determines eigenvectors by undoing the REDUC2 transformation.

      Discussion:

        REBAKB forms the eigenvectors of a generalized symmetric eigensystem by
        back transforming those of the derived symmetric matrix determined
        by REDUC2.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        27 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double B[N,N], contains information about the similarity
        transformation (Cholesky decomposition) used in the reduction by REDUC2
        in its strict lower triangle.

        Input, double DL[N], further information about the
        transformation.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double Z[N,M].  On input, the eigenvectors to be
        back transformed in its first M columns.  On output, the transformed
        eigenvectors.
    */
    {
      int i;
      int j;
      int k;
      double t;

      for ( j = 0; j < m; j++ )
      {
        for ( i = n - 1; 0 <= i; i-- )
        {
          t = dl[i] * z[i+j*n];
          for ( k = 0; k < i; k++ )
          {
            t = t + b[i+k*n] * z[k+j*n];
          }
          z[i+j*n] = t;
        }
      }

      return;
    }
    /******************************************************************************/

    int reduc ( int n, double a[], double b[], double dl[] )

    /******************************************************************************/
    /*
      Purpose:

        REDUC reduces the eigenvalue problem A*x=lambda*B*x to A*x=lambda*x.

      Discussion:

        REDUC reduces the generalized symmetric eigenproblem
        a x=(lambda) b x, where B is positive definite, to the standard
        symmetric eigenproblem using the Cholesky factorization of B.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        25 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrices A and B.  If the
        Cholesky factor L of B is already available, N should be prefixed with a
        minus sign.

        Input/output, double A[N,N].  On input, A contains a real
        symmetric matrix.  Only the full upper triangle of the matrix need be
        supplied.  On output, A contains in its full lower triangle the full lower
        triangle of the symmetric matrix derived from the reduction to the
        standard form.  The strict upper triangle of a is unaltered.

        Input/output, double B[N,N].  On input, the real symmetric
        input matrix.  Only the full upper triangle of the matrix need be supplied.
        If N is negative, the strict lower triangle of B contains, instead, the
        strict lower triangle of its Cholesky factor L.  In any case, on output,
        B contains in its strict lower triangle the strict lower triangle of
        its Cholesky factor L.  The full upper triangle of B is unaltered.

        Input/output, double DL[N].  If N is negative, then the DL
        contains the diagonal elements of L on input.  In any case, DL will contain
        the diagonal elements of L on output,

        Output, int REDUC, error flag.
        0, for normal return,
        7*N+1, if B is not positive definite.
    */
    {
      int i;
      int ierr;
      int j;
      int k;
      int nn;
      double x;
      double y = 0.0;

      ierr = 0;
      nn = abs ( n );
    /*
      Form L in the arrays B and DL.
    */
      for ( i = 0; i < n; i++ )
      {
        for ( j = i; j < n; j++ )
        {
          x = b[i+j*n];

          for ( k = 0; k < i; k++ )
          {
            x = x - b[i+k*n] * b[j+k*n];
          }

          if ( j == i )
          {
            if ( x <= 0.0 )
            {
              printf ( "\n" );
              printf ( "REDUC - Fatal error!\n" );
              printf ( "   The matrix is not positive definite.\n" );
              ierr = 7 * n + 1;
              return ierr;
            }

            y = sqrt ( x );
            dl[i] = y;
          }
          else
          {
            b[j+i*n] = x / y;
          }
        }
      }
    /*
      Form the transpose of the upper triangle of INV(L)*A
      in the lower triangle of the array A.
    */
      for ( i = 0; i < nn; i++ )
      {
        y = dl[i];

        for ( j = i; j < nn; j++ )
        {
          x = a[i+j*n];

          for ( k = 0; k < i; k++ )
          {
            x = x - b[i+k*n] * a[j+k*n];
          }
          a[j+i*n] = x / y;
        }
      }
    /*
      Pre-multiply by INV(L) and overwrite.
    */
      for ( j = 0; j < nn; j++ )
      {
        for ( i = j; i < nn; i++ )
        {
          x = a[i+j*n];

          for ( k = j; k < i; k++ )
          {
            x = x - a[k+j*n] * b[i+k*n];
          }

          for ( k = 0; k < j; k++ )
          {
            x = x - a[j+k*n] * b[i+k*n];
          }
          a[i+j*n] = x / dl[i];
        }
      }

      return ierr;
    }
    /******************************************************************************/

    int reduc2 ( int n, double a[], double b[], double dl[] )

    /******************************************************************************/
    /*
      Purpose:

        REDUC2 reduces the eigenvalue problem A*B*x=lamdba*x to A*x=lambda*x.

      Discussion:

        REDUC2 reduces the generalized symmetric eigenproblems
        a*b*x=lambda*x or b*a*y=lambda*y, where B is positive definite,
        to the standard symmetric eigenproblem using the Cholesky
        factorization of B.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        25 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrices A and B.  If the
        Cholesky factor L of B is already available, N should be prefixed with a
        minus sign.

        Input/output, double A[N,N].  On input, A contains a real
        symmetric matrix.  Only the full upper triangle of the matrix need be
        supplied.  On output, A contains in its full lower triangle the full lower
        triangle of the symmetric matrix derived from the reduction to the
        standard form.  The strict upper triangle of a is unaltered.

        Input/output, double B[N,N].  On input, the real symmetric
        input matrix.  Only the full upper triangle of the matrix need be supplied.
        If N is negative, the strict lower triangle of B contains, instead, the
        strict lower triangle of its Cholesky factor L.  In any case, on output,
        B contains in its strict lower triangle the strict lower triangle of
        its Cholesky factor L.  The full upper triangle of B is unaltered.

        Input/output, double DL[N].  If N is negative, then the DL
        contains the diagonal elements of L on input.  In any case, DL will contain
        the diagonal elements of L on output,

        Output, int REDUC2, error flag.
        0, for normal return,
        7*N+1, if B is not positive definite.
    */
    {
      int i;
      int ierr;
      int j;
      int k;
      int nn;
      double x;
      double y = 0.0;

      ierr = 0;
      nn = abs ( n );
    /*
      Form L in the arrays B and DL.
    */
      for ( i = 0; i < n; i++ )
      {
        for ( j = i; j < n; j++ )
        {
          x = b[i+j*nn];

          for ( k = 0; k < i; k++ )
          {
            x = x - b[i+k*nn] * b[j+k*nn];
          }

          if ( j == i )
          {
            if ( x <= 0.0 )
            {
              printf ( "\n" );
              printf ( "REDUC2 - Fatal error!\n" );
              printf ( "  The matrix is not positive definite.\n" );
              ierr = 7 * n + 1;
              return ierr;
            }

            y = sqrt ( x );
            dl[i] = y;
          }
          else
          {
            b[j+i*nn] = x / y;
          }
        }
      }
    /*
      Form the lower triangle of A*L in the lower triangle of A.
    */
      for ( i = 0; i < nn; i++ )
      {
        for ( j = 0; j <= i; j++ )
        {
          x = a[j+i*nn] * dl[j];

          for ( k = j + 1; k <= i; k++ )
          {
            x = x + a[k+i*nn] * b[k+j*nn];
          }

          for ( k = i + 1; k < nn; k++ )
          {
            x = x + a[i+k*nn] * b[k+j*nn];
          }
          a[i+j*nn] = x;
        }
      }
    /*
      Pre-multiply by L' and overwrite.
    */
      for ( i = 0; i < nn; i++ )
      {
        y = dl[i];

        for ( j = 0; j <= i; j++ )
        {
          x = y * a[i+j*nn];

          for ( k = i + 1; k < nn; k++ )
          {
            x = x + a[k+j*nn] * b[k+i*nn];
          }
          a[i+j*nn] = x;
        }
      }

      return ierr;
    }
    /******************************************************************************/


    /******************************************************************************/


    /******************************************************************************/
    /******************************************************************************/

    /******************************************************************************/

    /******************************************************************************/

    int rs ( int n, double a[], double w[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RS computes eigenvalues and eigenvectors of real symmetric matrix.

      Discussion:

        RS calls the recommended sequence of
        function from the eigensystem subroutine package (eispack)
        to find the eigenvalues and eigenvectors (if desired)
        of a real symmetric matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        10 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double A[N*N], the real symmetric matrix.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double W[N], the eigenvalues in ascending order.

        Output, double Z[N*N], contains the eigenvectors, if MATZ
        is true.

        Output, int RS, is set equal to an error
        completion code described in the documentation for TQLRAT and TQL2.
        The normal completion code is zero.
    */
    {
      double *fv1;
      double *fv2;
      int ierr;

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );

      if ( ! matz )
      {
        fv2 = ( double * ) malloc ( n * sizeof ( double ) );

        tred1 ( n, a, w, fv1, fv2 );

        ierr = tqlrat ( n, w, fv2 );

        if ( ierr != 0 )
        {
          free ( fv2 );

          printf ( "\n" );
          printf ( "RS - Fatal error!\n" );
          printf ( "  Error return from TQLRAT!\n" );
          return ierr;
        }

        free ( fv2 );
      }
      else
      {
        fv1 = ( double * ) malloc ( n * sizeof ( double ) );

        tred2 ( n, a, w, fv1, z );

        ierr = tql2 ( n, w, fv1, z );

        if ( ierr != 0 )
        {
          free ( fv1 );

          printf ( "\n" );
          printf ( "RS - Fatal error!\n" );
          printf ( "  Error return from TQL2!\n" );
          return ierr;
        }
      }

      free ( fv1 );

      return ierr;
    }
    /******************************************************************************/

    int rsb ( int n, int mb, double a[], double w[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RSB computes eigenvalues and eigenvectors of a real symmetric band matrix.

      Discussion:

        RSB calls the recommended sequence of
        functions from the eigensystem subroutine package (eispack)
        to find the eigenvalues and eigenvectors (if desired)
        of a real symmetric band matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        12 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int MB, the half band width of the matrix,
        defined as the number of adjacent diagonals, including the principal
        diagonal, required to specify the non-zero portion of the lower triangle
        of the matrix.

        Input, double A[N*MB], contains the lower triangle of the real
        symmetric band matrix.  Its lowest subdiagonal is stored in the last N+1-MB
        positions of the first column, its next subdiagonal in the last
        N+2-MB positions of the second column, further subdiagonals similarly,
        and finally its principal diagonal in the N positions of the last
        column.  Contents of storages not part of the matrix are arbitrary.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double W[N], the eigenvalues in ascending order.

        Output, double Z[N*N], contains the eigenvectors, if MATZ
        is true.

        Output, int BANDR, is set to an error
        completion code described in the documentation for TQLRAT and TQL2.
        The normal completion code is zero.
    */
    {
      double *fv1;
      double *fv2;
      int ierr;

      if ( mb <= 0 )
      {
        printf ( "\n" );
        printf ( "RSB - Fatal error!\n" );
        printf ( "  MB <= 0!\n" );
        ierr = 12 * n;
        return ierr;
      }

      if ( n < mb )
      {
        printf ( "\n" );
        printf ( "RSB - Fatal error!\n" );
        printf ( "  N < MB!\n" );
        ierr = 12 * n;
        return ierr;
      }

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );

      if ( ! matz )
      {
        fv2 = ( double * ) malloc ( n * sizeof ( double ) );

        bandr ( n, mb, a, w, fv1, fv2, matz, z );

        ierr = tqlrat ( n, w, fv2 );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          printf ( "\n" );
          printf ( "RSB - Fatal error!\n" );
          printf ( "  Error return from TQLRAT!\n" );
          return ierr;
        }

        free ( fv2 );
      }
      else
      {
        bandr ( n, mb, a, w, fv1, fv1, matz, z );

        ierr = tql2 ( n, w, fv1, z );

        if ( ierr != 0 )
        {
          free ( fv1 );

          printf ( "\n" );
          printf ( "RSB - Fatal error!\n" );
          printf ( "  Error return from TQLRAT!\n" );
          return ierr;
        }
      }

      free ( fv1 );

      return ierr;
    }
    /******************************************************************************/

    int rsg ( int n, double a[], double b[], double w[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RSG computes eigenvalues/vectors, A*x=lambda*B*x, A symmetric, B pos-def.

      Discussion:

        RSG calls the recommended sequence of EISPACK functions
        to find the eigenvalues and eigenvectors (if desired)
        for the real symmetric generalized eigenproblem  a x = (lambda) b x.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        24 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrices A and B.

        Input, double A[N,N], contains a real symmetric matrix.

        Input, double B[N,N], contains a positive definite real
        symmetric matrix.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double W[N], the eigenvalues in ascending order.

        Output, double Z[N,N], contains the eigenvectors, if MATZ
        is true.

        Output, int RSG, is set to an error
        completion code described in the documentation for TQLRAT and TQL2.
        The normal completion code is zero.
    */
    {
      double *fv1;
      double *fv2;
      int ierr;

      fv2 = ( double * ) malloc ( n * sizeof ( double ) );

      ierr = reduc ( n, a, b, fv2 );

      if ( ierr != 0 )
      {
        free ( fv2 );
        printf ( "\n" );
        printf ( "RSG - Fatal error!\n" );
        printf ( "  Error return from REDUC.\n" );
        return ierr;
      }

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );

      if ( ! matz )
      {
        tred1 ( n, a, w, fv1, fv2 );

        ierr = tqlrat ( n, w, fv2 );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          printf ( "\n" );
          printf ( "RSG - Fatal error!\n" );
          printf ( "  Error return from TQLRAT!\n" );
          return ierr;
        }
      }
      else
      {
        tred2 ( n, a, w, fv1, z );

        ierr = tql2 ( n, w, fv1, z );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          printf ( "\n" );
          printf ( "RSG - Fatal error!\n" );
          printf ( "  Error return from TQL2!\n" );
          return ierr;
        }

        rebak ( n, b, fv2, n, z );
      }

      free ( fv1 );
      free ( fv2 );

      return ierr;
    }
    /******************************************************************************/

    int rsgab ( int n, double a[], double b[], double w[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RSGAB computes eigenvalues/vectors, A*B*x=lambda*x, A symmetric, B pos-def.

      Discussion:

        RSGAB calls the recommended sequence of EISPACK functions
        to find the eigenvalues and eigenvectors (if desired)
        for the real symmetric generalized eigenproblem
          A * B * x = lambda * x.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        26 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrices A and B.

        Input, double A[N,N], contains a real symmetric matrix.

        Input, double B[N,N], contains a positive definite real
        symmetric matrix.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double W[N], the eigenvalues in ascending order.

        Output, double Z[N,N], contains the eigenvectors, if MATZ
        is true.

        Output, int RSGAB, is set to an error
        completion code described in the documentation for TQLRAT and TQL2.
        The normal completion code is zero.
    */
    {
      double *fv1;
      double *fv2;
      int ierr;

      fv2 = ( double * ) malloc ( n * sizeof ( double ) );

      ierr = reduc2 ( n, a, b, fv2 );

      if ( ierr != 0 )
      {
        free ( fv2 );
        printf ( "\n" );
        printf ( "RSGAB - Fatal error!\n" );
        printf ( "  Error return from REDUC2!\n" );
        return ierr;
      }

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );

      if ( ! matz )
      {
        tred1 ( n, a, w, fv1, fv2 );

        ierr = tqlrat ( n, w, fv2 );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          printf ( "\n" );
          printf ( "RSGAB - Fatal error!\n" );
          printf ( "  Error return from TQLRAT!\n" );
          return ierr;
        }
      }
      else
      {
        tred2 ( n, a, w, fv1, z );

        ierr = tql2 ( n, w, fv1, z );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          printf ( "\n" );
          printf ( "RSB - Fatal error!\n" );
          printf ( "  Error return from TQL2!\n" );
          return ierr;
        }

        rebak ( n, b, fv2, n, z );
      }
    /*
      Free memory.
    */
      free ( fv1 );
      free ( fv2 );

      return ierr;
    }
    /******************************************************************************/

    int rsgba ( int n, double a[], double b[], double w[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RSGBA computes eigenvalues/vectors, B*A*x=lambda*x, A symmetric, B pos-def.

      Discussion:

        RSGBA calls the recommended sequence of EISPACK functions
        to find the eigenvalues and eigenvectors (if desired)
        for the real symmetric generalized eigenproblem:

          B * A * x = lambda * x

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        27 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrices A and B.

        Input, double A[N,N], a real symmetric matrix.

        Input, double B[N,N], a positive definite symmetric matrix.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double W[N], the eigenvalues in ascending order.

        Output, double Z[N,N], contains the eigenvectors, if MATZ
        is true.

        Output, int RSGBA, is set to an error
        completion code described in the documentation for TQLRAT and TQL2.
        The normal completion code is zero.
    */
    {
      double *fv1;
      double *fv2;
      int ierr;

      fv2 = ( double * ) malloc ( n * sizeof ( double ) );

      ierr = reduc2 ( n, a, b, fv2 );

      if ( ierr != 0 )
      {
        free ( fv2 );
        printf ( "\n" );
        printf ( "RSGBA - Fatal error!\n" );
        printf ( "  Error return from REDUC2!\n" );
        return ierr;
      }

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );

      if ( ! matz )
      {
        tred1 ( n, a, w, fv1, fv2 );

        ierr = tqlrat ( n, w, fv2 );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          printf ( "\n" );
          printf ( "RSGBA - Fatal error!\n" );
          printf ( "  Error return from TQLRAT!\n" );
          return ierr;
        }
      }
      else
      {
        tred2 ( n, a, w, fv1, z );

        ierr = tql2 ( n, w, fv1, z );

        if ( ierr != 0 )
        {
          free ( fv1 );
          free ( fv2 );
          printf ( "\n" );
          printf ( "RSGBA - Fatal error!\n" );
          printf ( "  Error return from TQL2!\n" );
          return ierr;
        }

        rebakb ( n, b, fv2, n, z );
      }
    /*
      Free memory.
    */
      free ( fv1 );
      free ( fv2 );

      return ierr;
    }
    /******************************************************************************/

    int rsm ( int n, double a[], double w[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RSM computes eigenvalues, some eigenvectors, real symmetric matrix.

      Discussion:

        RSM calls the recommended sequence of EISPACK routines
        to find all of the eigenvalues and some of the eigenvectors
        of a real symmetric matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        09 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double A[N,N], the symmetric matrix.

        Output, double W[N], the eigenvalues in ascending order.

        Input, int M, specifies the number of eigenvectors to
        compute.

        Output, double Z[N,M], contains the orthonormal eigenvectors
        associated with the first M eigenvalues.

        Output, int RSM, is set to an error
        completion code described in the documentation for TQLRAT, IMTQLV and
        TINVIT.  The normal completion code is zero.
    */
    {
      double *fwork1;
      double *fwork2;
      double *fwork3;
      int ierr;
      int *iwork;

      fwork1 = ( double * ) malloc ( n * sizeof ( double ) );
      fwork2 = ( double * ) malloc ( n * sizeof ( double ) );

      if ( m <= 0 )
      {
        tred1 ( n, a, w, fwork1, fwork2 );

        ierr = tqlrat ( n, w, fwork2 );

        if ( ierr != 0 )
        {
          free ( fwork1 );
          free ( fwork2 );
          printf ( "\n" );
          printf ( "RSM - Fatal error!\n" );
          printf ( "  Error return from TQLRAT!\n" );
          return ierr;
        }
      }
      else
      {
        fwork3 = ( double * ) malloc ( n * sizeof ( double ) );

        tred1 ( n, a, fwork1, fwork2, fwork3 );

        iwork = ( int * ) malloc ( n * sizeof ( int ) );

        ierr = imtqlv ( n, fwork1, fwork2, fwork3, w, iwork );

        if ( ierr != 0 )
        {
          free ( fwork1 );
          free ( fwork2 );
          free ( fwork3 );
          free ( iwork );
          printf ( "\n" );
          printf ( "RSM - Fatal error!\n" );
          printf ( "  Error return from IMTQLV!\n" );
          return ierr;
        }

        ierr = tinvit ( n, fwork1, fwork2, fwork3, m, w, iwork, z );

        if ( ierr != 0 )
        {
          free ( fwork1 );
          free ( fwork2 );
          free ( fwork3 );
          free ( iwork );
          printf ( "\n" );
          printf ( "RSM - Fatal error!\n" );
          printf ( "  Error return from TINVIT!\n" );
          return ierr;
        }

        trbak1 ( n, a, fwork2, m, z );
      }

      free ( fwork1 );
      free ( fwork2 );

      return ierr;
    }
    /******************************************************************************/

    int rsp ( int n, int nv, double a[], double w[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RSP computes eigenvalues and eigenvectors of real symmetric packed matrix.

      Discussion:

        RSP calls the recommended sequence of EISPACK routines
        to find the eigenvalues and eigenvectors (if desired)
        of a real symmetric packed matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        27 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int NV, the dimension of the array A, which
        must be at least (N*(N+1))/2.

        Input, double A[NV], contains the lower triangle of the
        real symmetric packed matrix stored row-wise.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double W[N], the eigenvalues in ascending order.

        Output, double Z[N,N], contains the eigenvectors, if MATZ is
        true.

        Output, int RSP, is set to an error
        completion code described in the documentation for TQLRAT and TQL2.
        The normal completion code is zero.
    */
    {
      double *fv1;
      double *fv2;
      //int i;
      int ierr;

      if ( ( n * ( n + 1 ) ) / 2 > nv )
      {
        ierr = 20 * n;
        printf ( "\n" );
        printf ( "RSP - Fatal error!\n" );
        printf ( "  NV is too small!\n" );
        return ierr;
      }

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );
      fv2 = ( double * ) malloc ( n * sizeof ( double ) );

      tred3 ( n, nv, a, w, fv1, fv2 );

      if ( ! matz )
      {
        ierr = tqlrat ( n, w, fv2 );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RSP - Fatal error!\n" );
          printf ( "  Error return from TQLRAT.\n" );
          free ( fv1 );
          free ( fv2 );
          return ierr;
        }
      }
      else
      {
        r8mat_identity ( n, z );

        ierr = tql2 ( n, w, fv1, z );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RSP - Fatal error!\n" );
          printf ( "  Error return from TQL2.\n" );
          free ( fv1 );
          free ( fv2 );
          return ierr;
        }

        trbak3 ( n, nv, a, n, z );
      }

      free ( fv1 );
      free ( fv2 );

      return ierr;
    }
    /******************************************************************************/

    int rspp ( int n, int nv, double a[], double w[], bool matz, double z[], int m,
      bool type )

    /******************************************************************************/
    /*
      Purpose:

        RSPP computes some eigenvalues/vectors, real symmetric packed matrix.

      Discussion:

        RSPP calls the appropriate routines for the following problem:

        Given a symmetric matrix A, which is stored in a packed mode, find
        the M smallest or largest eigenvalues, and corresponding eigenvectors.

        The routine RSP returns all eigenvalues and eigenvectors.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        10 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of A, the number of rows and
        columns in the original matrix.

        Input, int NV, is the of the array A as specified in the
        calling program.  NV must not be less than N*(N+1)/2.

        Input, double A[(N*(N+1))/2], on input the lower triangle of the
        real symmetric matrix, stored row-wise in the vector,
        in the order A(1,1), / A(2,1), A(2,2), / A(3,1), A(3,2), A(3,3)/
        and so on.

        Output, double W[M], the eigenvalues requested.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double Z[N,M], the eigenvectors.

        Input, int M, the number of eigenvalues to be found.

        Input, bool TYPE, is true if the smallest eigenvalues
        are to be found, or false if the largest ones are sought.

        Output, int RSPP, error flag from RATQR.
        0 on normal return.
        nonzero, the algorithm broke down while computing an eigenvalue.
    */
    {
      double *bd;
      double eps1;
      int idef;
      int ierr;
      int *iwork;
      double *work1;
      double *work2;
      double *work3;
    /*
      IDEF =
        -1 if the matrix is known to be negative definite,
        +1 if the matrix is known to be positive definite, or
        0 otherwise.
    */
      idef = 0;
    /*
      Reduce to symmetric tridiagonal form.
    */
      work1 = ( double * ) malloc ( n * sizeof ( double ) );
      work2 = ( double * ) malloc ( n * sizeof ( double ) );
      work3 = ( double * ) malloc ( n * sizeof ( double ) );

      tred3 ( n, nv, a, work1, work2, work3 );
    /*
      Find the eigenvalues.
    */
      eps1 = 0.0;

      iwork = ( int * ) malloc ( n * sizeof ( int ) );
      bd = ( double * ) malloc ( n * sizeof ( double ) );

      ierr = ratqr ( n, eps1, work1, work2, work3, m, w, iwork, bd, type, idef );

      if ( ierr != 0 )
      {
        free ( bd );
        free ( iwork );
        free ( work1 );
        free ( work2 );
        free ( work3 );
        printf ( "\n" );
        printf ( "RSPP - Fatal error!\n" );
        printf ( "  Error return from RATQR.\n" );
        return ierr;
      }
    /*
      Find eigenvectors for the first M eigenvalues.
    */
      if ( matz )
      {
        ierr = tinvit ( n, work1, work2, work3, m, w, iwork, z );

        if ( ierr != 0 )
        {
          free ( bd );
          free ( iwork );
          free ( work1 );
          free ( work2 );
          free ( work3 );
          printf ( "\n" );
          printf ( "RSPP - Fatal error!\n" );
          printf ( "  Error return from TINVIT.\n" );
          return ierr;
        }
    /*
      Reverse the transformation.
    */
        trbak3 ( n, nv, a, m, z );
      }

      free ( bd );
      free ( iwork );
      free ( work1 );
      free ( work2 );
      free ( work3 );

      return ierr;
    }
    /******************************************************************************/

    int rst ( int n, double w[], double e[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RST computes eigenvalues/vectors, real symmetric tridiagonal matrix.

      Discussion:

        RST calls the recommended sequence of EISPACK routines
        to find the eigenvalues and eigenvectors (if desired)
        of a real symmetric tridiagonal matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        04 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double W[N].  On input, the diagonal elements
        of the real symmetric tridiagonal matrix.  On output, the eigenvalues in
        ascending order.

        Input, double E[N], the subdiagonal elements of the matrix in
        E(2:N).  E(1) is arbitrary.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double Z[N,N], contains the eigenvectors, if MATZ
        is true.

        Output, int RST_TEST, is set to an error
        completion code described in the documentation for IMTQL1 and IMTQL2.
        The normal completion code is zero.
    */
    {
      //int i;
      int ierr;

      if ( ! matz )
      {
        ierr = imtql1 ( n, w, e );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RST - Fatal error!\n" );
          printf ( "  Error return from IMTQL1.\n" );
          return ierr;
        }
      }
      else
      {
        r8mat_identity ( n, z );

        ierr = imtql2 ( n, w, e, z );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RST - Fatal error!\n" );
          printf ( "  Error return from IMTQL2.\n" );
          return ierr;
        }
      }

      return ierr;
    }
    /******************************************************************************/

    int rt ( int n, double a[], double w[], bool matz, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        RT computes eigenvalues/vectors, real sign-symmetric tridiagonal matrix.

      Discussion:

        RT calls the recommended sequence of EISPACK routines
        to find the eigenvalues and eigenvectors (if desired)
        of a special real tridiagonal matrix.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double A[N,N], contains the special real tridiagonal
        matrix in its first three columns.  The subdiagonal elements are stored
        in the last N-1 positions of the first column, the diagonal elements
        in the second column, and the superdiagonal elements in the first N-1
        positions of the third column.  Elements A(1,1) and A(N,3) are arbitrary.

        Input, bool MATZ, is false if only eigenvalues are desired,
        and true if both eigenvalues and eigenvectors are desired.

        Output, double W[N], the eigenvalues in ascending order.

        Output, double Z[N,N], contains the eigenvectors, if MATZ
        is true.

        Output, int RT, is set to an error
        completion code described in the documentation for IMTQL1 and IMTQL2.
        The normal completion code is zero.
    */
    {
      double *fv1;
      int ierr;

      fv1 = ( double * ) malloc ( n * sizeof ( double ) );

      if ( ! matz )
      {
        ierr = figi ( n, a, w, fv1, fv1 );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RT - Fatal error!\n" );
          printf ( "  Error return from FIGI.\n" );
          return ierr;
        }

        ierr = imtql1 ( n, w, fv1 );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RT - Fatal error!\n" );
          printf ( "  Error return from IMTQL1.\n" );
          return ierr;
        }
      }
      else
      {
        ierr = figi2 ( n, a, w, fv1, z );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RT - Fatal error!\n" );
          printf ( "  Error return from FIGI2.\n" );
          return ierr;
        }

        ierr = imtql2 ( n, w, fv1, z );

        if ( ierr != 0 )
        {
          printf ( "\n" );
          printf ( "RT - Fatal error!\n" );
          printf ( "  Error return from IMTQL2.\n" );
          return ierr;
        }
      }

      free ( fv1 );

      return ierr;
    }
    /******************************************************************************/

    int svd ( int m, int n, double a[], double w[], bool matu, double u[],
      bool matv, double v[] )

    /******************************************************************************/
    /*
      Purpose:

        SVD computes the singular value decomposition for a real matrix.

      Discussion:

        SVD determines the singular value decomposition

          A = U * S * V'

        of a real M by N rectangular matrix.  Householder bidiagonalization
        and a variant of the QR algorithm are used.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Golub, Christian Reinsch,
        Numerische Mathematik,
        Volume 14, 1970, pages 403-420.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int M, the number of rows of A and U.

        Input, int N, the number of columns of A and U, and
        the order of V.

        Input, double A[M,N], the M by N matrix to be decomposed.

        Output, double W[N], the singular values of A.  These are the
        diagonal elements of S.  They are unordered.  If an error break; is
        made, the singular values should be correct for indices
        IERR+1, IERR+2,..., N.

        Input, bool MATU, should be set to TRUE if the U matrix in the
        decomposition is desired, and to FALSE otherwise.

        Output, double U[M,N], contains the matrix U, with orthogonal
        columns, of the decomposition, if MATU has been set to TRUE.  Otherwise
        U is used as a temporary array.  U may coincide with A.
        If an error break; is made, the columns of U corresponding
        to indices of correct singular values should be correct.

        Input, bool MATV, should be set to TRUE if the V matrix in the
        decomposition is desired, and to FALSE otherwise.

        Output, double V[N,N], the orthogonal matrix V of the
        decomposition if MATV has been set to TRUE.  Otherwise V is not referenced.
        V may also coincide with A if U is not needed.  If an error
        break; is made, the columns of V corresponding to indices of
        correct singular values should be correct.

        Output, int SVD, error flag.
        0, for normal return,
        K, if the K-th singular value has not been determined after 30 iterations.
    */
    {
      double c;
      double f;
      double g;
      double h;
      int i;
      int ierr;
      int ii;
      int its;
      int j;
      int k;
      int l;
      int mn;
      double *rv1;
      double s;
      double scale;
      bool skip;
      double tst1;
      double tst2;
      double x;
      double y;
      double z;

      rv1 = ( double * ) malloc ( n * sizeof ( double ) );

      ierr = 0;

      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          u[i+j*m] = a[i+j*m];
        }
      }
    /*
      Householder reduction to bidiagonal form.
    */
      g = 0.0;
      scale = 0.0;
      x = 0.0;

      for ( i = 0; i < n; i++ )
      {
        rv1[i] = scale * g;
        g = 0.0;
        s = 0.0;
        scale = 0.0;

        if ( i <= m )
        {
          scale = 0.0;
          for ( ii = i; ii < m; ii++ )
          {
            scale = scale + fabs ( u[ii+i*m] );
          }

          if ( scale != 0.0 )
          {
            for ( ii = i; ii < m; ii++ )
            {
              u[ii+i*m] = u[ii+i*m] / scale;
            }
            s = 0.0;
            for ( ii = i; ii < m; ii++ )
            {
              s = s + u[ii+i*m] * u[ii+i*m];
            }
            f = u[i+i*m];
            g = - sqrt ( s ) * r8_sign ( f );
            h = f * g - s;
            u[i+i*m] = f - g;

            for ( j = i + 1; j < n; j++ )
            {
              s = 0.0;
              for ( k = i; k < m; k++ )
              {
                s = s + u[k+i*m] * u[k+j*m];
              }
              for ( k = i; k < m; k++ )
              {
                u[k+j*m] = u[k+j*m] + s * u[k+i*m] / h;
              }
            }
            for ( k = i; k < m; k++ )
            {
              u[k+i*m] = scale * u[k+i*m];
            }
          }
        }

        w[i] = scale * g;
        g = 0.0;
        s = 0.0;
        scale = 0.0;

        if ( i < m && i != n - 1 )
        {
          scale = 0.0;
          for ( k = i + 1; k < n; k++ )
          {
            scale = scale + fabs ( u[i+k*m] );
          }
          if ( scale != 0.0 )
          {
            for ( k = i + 1; k < n; k++ )
            {
              u[i+k*m] = u[i+k*m] / scale;
            }
            s = 0.0;
            for ( k = i + 1; k < n; k++ )
            {
              s = s + u[i+k*m] * u[i+k*m];
            }
            f = u[i+(i+1)*m];
            g = - sqrt ( s ) * r8_sign ( f );
            h = f * g - s;
            u[i+(i+1)*m] = f - g;
            for ( ii = i + 1; ii < n; ii++ )
            {
              rv1[ii] = u[i+ii*m] / h;
            }
            for ( j = i + 1; j < m; j++ )
            {
              s = 0.0;
              for ( k = i + 1; k < n; k++ )
              {
                s = s + u[j+k*m] * u[i+k*m];
              }
              for ( k = i + 1; k < n; k++ )
              {
                u[j+k*m] = u[j+k*m] + s * rv1[k];
              }
            }
            for ( k = i + 1; k < n; k++ )
            {
              u[i+k*m] = u[i+k*m] * scale;
            }
          }
        }

        x = r8_max ( x, fabs ( w[i] ) + fabs ( rv1[i] ) );
      }
    /*
      Accumulation of right-hand transformations.
    */
      if ( matv )
      {
        for ( i = n - 1; 0 <= i; i-- )
        {
          if ( i < n - 1 )
          {
            if ( g != 0.0 )
            {
              for ( k = i + 1; k < n; k++ )
              {
                v[k+i*n] = ( u[i+k*m] / u[i+(i+1)*m] ) / g;
              }
              for ( j = i + 1; j < n; j++ )
              {
                s = 0.0;
                for ( k = i + 1; k < n; k++ )
                {
                  s = s + u[i+k*m] * v[k+j*n];
                }
                for ( k = i + 1; k < n; k++ )
                {
                  v[k+j*n] = v[k+j*n] + s * v[k+i*n];
                }
              }
            }
            for ( k = i + 1; k < n; k++ )
            {
              v[i+k*n] = 0.0;
              v[k+i*n] = 0.0;
            }
          }
          v[i+i*n] = 1.0;
          g = rv1[i];
        }
      }
    /*
      Accumulation of left-hand transformations.
    */
      if ( matu )
      {
        mn = i4_min ( m, n );

        for ( i = mn - 1; 0 <= i; i-- )
        {
          g = w[i];

          if ( i != n - 1 )
          {
            for ( k = i + 1; k < n; k++ )
            {
              u[i+k*m] = 0.0;
            }
          }

          if ( g != 0.0 )
          {
            if ( i != mn - 1 )
            {
              for ( j = i + 1; j < n; j++ )
              {
                s = 0.0;
                for ( k = i + 1; k < m; k++ )
                {
                  s = s + u[k+i*m] * u[k+j*m];
                }
                f = ( s / u[i+i*m] ) / g;
                for ( k = i; k < m; k++ )
                {
                  u[k+j*m] = u[k+j*m] + f * u[k+i*m];
                }
              }
            }
            for ( k = i; k < m; k++ )
            {
              u[k+i*m] = u[k+i*m] / g;
            }
          }
          else
          {
            for ( k = i; k < m; k++ )
            {
              u[k+i*m] = 0.0;
            }
          }
          u[i+i*m] = u[i+i*m] + 1.0;
        }
      }
    /*
      Diagonalization of the bidiagonal form.
    */
      tst1 = x;

      for ( k = n - 1; 0 <= k; k-- )
      {
        its = 0;
    /*
      Test for splitting.
    */
        while ( true )
        {
          skip = false;

          for ( l = k; 1 <= l; l-- )
          {
            tst2 = tst1 + fabs ( rv1[l] );

            if ( tst2 == tst1 )
            {
              skip = true;
              break;
            }

            tst2 = tst1 + fabs ( w[l-1] );

            if ( tst2 == tst1 )
            {
              break;
            }
          }
    /*
      Cancellation of rv1[l] if L greater than 1.
    */
          if ( ! skip )
          {
            c = 0.0;
            s = 1.0;

            for ( i = l; i <= k; i++ )
            {
              f = s * rv1[i];
              rv1[i] = c * rv1[i];
              tst2 = tst1 + fabs ( f );

              if ( tst2 == tst1 )
              {
                break;
              }

              g = w[i];
              h = pythag ( f, g );
              w[i] = h;
              c = g / h;
              s = - f / h;

              if ( matu )
              {
                for ( j = 0; j < m; j++ )
                {
                  y = u[j+(l-1)*m];
                  z = u[j+i*m];
                  u[j+(l-1)*m] =  y * c + z * s;
                  u[j+i*m] =    - y * s + z * c;
                }
              }
            }
          }

          z = w[k];
    /*
      Convergence.
    */
          if ( l == k )
          {
            if ( z <= 0.0 )
            {
              w[k] = - z;
              if ( matv )
              {
                for ( j = 0; j < n; j++ )
                {
                  v[j+k*n] = - v[j+k*n];
                }
              }
            }
            break;
          }
    /*
      Shift from bottom 2 by 2 minor.
    */
          if ( 30 <= its )
          {
            free ( rv1 );
            ierr = k;
            return ierr;
          }

          its = its + 1;
          x = w[l];
          y = w[k-1];
          g = rv1[k-1];
          h = rv1[k];
          f = 0.5 * ( ( ( g + z ) / h ) * ( ( g - z ) / y ) + y / h - h / y );
          g = pythag ( f, 1.0 );
          f = x - ( z / x ) * z + ( h / x )
            * ( y / ( f + fabs ( g ) * r8_sign ( f ) ) - h );
    /*
      Next QR transformation.
    */
          c = 1.0;
          s = 1.0;

          for ( i = l + 1; i <= k; i++ )
          {
            g = rv1[i];
            y = w[i];
            h = s * g;
            g = c * g;
            z = pythag ( f, h );
            rv1[i-1] = z;
            c = f / z;
            s = h / z;
            f =   x * c + g * s;
            g = - x * s + g * c;
            h = y * s;
            y = y * c;

            if ( matv )
            {
              for ( j = 0; j < n; j++ )
              {
                x = v[j+(i-1)*n];
                z = v[j+i*n];
                v[j+(i-1)*n] =   x * c + z * s;
                v[j+i*n]     = - x * s + z * c;
              }
            }

            z = pythag ( f, h );
            w[i-1] = z;
    /*
      Rotation can be arbitrary if Z is zero.
    */
            if ( z != 0.0 )
            {
              c = f / z;
              s = h / z;
            }

            f =   c * g + s * y;
            x = - s * g + c * y;

            if ( matu )
            {
              for ( j = 0; j < m; j++ )
              {
                y = u[j+(i-1)*m];
                z = u[j+i*m];
                u[j+(i-1)*m] =   y * c + z * s;
                u[j+i*m]     = - y * s + z * c;
              }
            }
          }
          rv1[l] = 0.0;
          rv1[k] = f;
          w[k] = x;
        }
      }

      free ( rv1 );

      return ierr;
    }
    /******************************************************************************/

    void timestamp ( )

    /******************************************************************************/
    /*
      Purpose:

        TIMESTAMP prints the current YMDHMS date as a time stamp.

      Example:

        31 May 2001 09:45:54 AM

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        24 September 2003

      Author:

        John Burkardt

      Parameters:

        None
    */
    {
    # define TIME_SIZE 40

      static char time_buffer[TIME_SIZE];
      const struct tm *tm;
      time_t now;

      now = time ( nullptr );
      tm = localtime ( &now );

      strftime (time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

      fprintf ( stdout, "%s\n", time_buffer );

      return;
    # undef TIME_SIZE
    }
    /******************************************************************************/

    int tinvit ( int n, double d[], double e[], double e2[], int m, double w[],
      int ind[], double z[] )

    /******************************************************************************/
    /*
      Purpose:

        TINVIT computes eigenvectors from eigenvalues, real tridiagonal symmetric.

      Discussion:

        TINVIT finds eigenvectors of a tridiagonal symmetric matrix corresponding
        to specified eigenvalues using inverse iteration.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        12 February 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double D[N], the diagonal elements of the matrix.

        Input, double E[N], contains the subdiagonal elements of
        the input matrix in E(2:N).  E(1) is arbitrary.

        Input, double E2[N], contains the squares of the corresponding
        elements of E, with zeros corresponding to negligible elements of E.
        E(I) is considered negligible if it is not larger than the product of
        the relative machine precision and the sum of the magnitudes of D(I)
        and D(I-1).  E2(1) must contain 0.0 if the eigenvalues are in
        ascending order, or 2.0 if the eigenvalues are in descending order.
        If BISECT, TRIDIB, or IMTQLV has been used to find the eigenvalues,
        their output E2 array is exactly what is expected here.

        Input, int M, the number of specified eigenvalues.

        Input, double W[M], the eigenvalues.

        Input, int IND[M], the submatrix indices associated with
        the corresponding eigenvalues in W: 1 for eigenvalues belonging to the
        first submatrix from the top, 2 for those belonging to the second
        submatrix, and so on.

        Output, double Z[N,M], the associated set of orthonormal
        eigenvectors.  Any vector which fails to converge is set to zero.

        Output, int TINVIT, error flag.
        0, for normal return,
        -R, if the eigenvector corresponding to the R-th eigenvalue fails to
          converge in 5 iterations.
    */
    {
      double eps2;
      double eps3;
      double eps4;
      int group;
      int i;
      int ierr;
      int its;
      int j;
      int jj;
      int k;
      double norm;
      double order;
      int p;
      int q;
      int r;
      double *rv1;
      double *rv2;
      double *rv3;
      double *rv4;
      double *rv6;
      int s;
      int tag;
      double u;
      double uk;
      double v;
      double x0;
      double x1;
      double xu;

      ierr = 0;

      if ( m == 0 )
      {
        return ierr;
      }

      rv1 = ( double * ) malloc ( n * sizeof ( double ) );
      rv2 = ( double * ) malloc ( n * sizeof ( double ) );
      rv3 = ( double * ) malloc ( n * sizeof ( double ) );
      rv4 = ( double * ) malloc ( n * sizeof ( double ) );
      rv6 = ( double * ) malloc ( n * sizeof ( double ) );

      u = 0.0;
      x0 = 0.0;

      tag = -1;
      order = 1.0 - e2[0];
      q = -1;
    /*
      Establish and process next submatrix.
    */
      while ( true )
      {
        p = q + 1;

        for ( q = p; q < n; q++ )
        {
          if ( q == n - 1 )
          {
            break;
          }
          if ( e2[q+1] == 0.0 )
          {
            break;
          }
        }
    /*
      Find vectors by inverse iteration.
    */
        tag = tag + 1;
        s = 0;

        for ( r = 0; r < m; r++ )
        {
          if ( ind[r] != tag )
          {
            continue;
          }

          its = 1;
          x1 = w[r];
    /*
      Look for close or coincident roots.
    */
          if ( s != 0 )
          {
            if ( eps2 <= fabs ( x1 - x0 ) )
            {
              group = 0;
            }
            else
            {
              group = group + 1;
              if ( order * ( x1 - x0 ) <= 0.0 )
              {
                x1 = x0 + order * eps3;
              }
            }
          }
    /*
      Check for isolated root.
    */
          else
          {
            xu = 1.0;

            if ( p == q )
            {
              rv6[p] = 1.0;
              for ( k = 0; k < n; k++ )
              {
                z[k+r*n] = 0.0;
              }
              for ( k = p; k <= q; k++ )
              {
                z[k+r*n] = rv6[k] * xu;
              }
              x0 = x1;
              continue;
            }

            norm = fabs ( d[p] );

            for ( i = p + 1; i <= q; i++ )
            {
              norm = r8_max ( norm, fabs ( d[i] ) + fabs ( e[i] ) );
            }
    /*
      EPS2 is the criterion for grouping,
      EPS3 replaces zero pivots and equal roots are modified by EPS3,
      EPS4 is taken very small to avoid overflow.
    */
            eps2 = 0.001 * norm;
            eps3 = fabs ( norm ) * r8_epsilon ( );
            uk = q - p + 1;
            eps4 = uk * eps3;
            uk = eps4 / sqrt ( uk );
            s = p;
            group = 0;
          }
    /*
      Elimination with interchanges and initialization of vector.
    */
          v = 0.0;

          for ( i = p; i <= q; i++ )
          {
            rv6[i] = uk;

            if ( i == p )
            {
              u = d[i] - x1 - xu * v;
              if ( i != q )
              {
                v = e[i+1];
              }
            }
            else if ( fabs ( u ) <= fabs ( e[i] ) )
            {
              xu = u / e[i];
              rv4[i] = xu;
              rv1[i-1] = e[i];
              rv2[i-1] = d[i] - x1;
              rv3[i-1] = 0.0;
              if ( i != q )
              {
                rv3[i-1] = e[i+1];
              }
              u = v - xu * rv2[i-1];
              v = - xu * rv3[i-1];
            }
            else
            {
              xu = e[i] / u;
              rv4[i] = xu;
              rv1[i-1] = u;
              rv2[i-1] = v;
              rv3[i-1] = 0.0;

              u = d[i] - x1 - xu * v;
              if ( i != q )
              {
                v = e[i+1];
              }
            }
          }

          if ( u == 0.0 )
          {
            u = eps3;
          }

          rv1[q] = u;
          rv2[q] = 0.0;
          rv3[q] = 0.0;
    /*
      Back substitution.
    */
          while ( true )
          {
            for ( i = q; p <= i; i-- )
            {
              rv6[i] = ( rv6[i] - u * rv2[i] - v * rv3[i] ) / rv1[i];
              v = u;
              u = rv6[i];
            }
    /*
      Orthogonalize with respect to previous members of group.
    */
            j = r;

            for ( jj = 1; jj <= group; jj++ )
            {
              while ( true )
              {
                j = j - 1;

                if ( ind[j] == tag )
                {
                  break;
                }
              }

              xu = 0.0;
              for ( k = p; k <= q; k++ )
              {
                xu = xu + rv6[k] * z[k+j*n];
              }
              for ( k = p; k <= q; k++ )
              {
                rv6[k] = rv6[k] - xu * z[k+j*n];
              }
            }

            norm = 0.0;
            for ( k = p; k <= q; k++ )
            {
              norm = norm + fabs ( rv6[k] );
            }
    /*
      Normalize so that sum of squares is 1.
    */
            if ( 1.0 <= norm )
            {
              u = 0.0;
              for( i = p; i <= q; i++ )
              {
                u = pythag ( u, rv6[i] );
              }
              xu = 1.0 / u;

              for ( k = 0; k < n; k++ )
              {
                z[k+r*n] = 0.0;
              }
              for ( k = p; k <= q; k++ )
              {
                z[k+r*n] = rv6[k] * xu;
              }
              x0 = x1;
              break;
            }
    /*
      Set error: non-converged eigenvector.
    */
            else if ( 5 <= its )
            {
              ierr = - r;
              xu = 0.0;
              for ( k = 0; k < n; k++ )
              {
                z[k+r*n] = 0.0;
              }
              for ( k = p; k <= q; k++ )
              {
                z[k+r*n] = rv6[k] * xu;
              }
              x0 = x1;
              break;
            }
            else
            {
              if ( norm == 0.0 )
              {
                rv6[s] = eps4;
                s = s + 1;
                if ( q < s )
                {
                  s = p;
                }
              }
              else
              {
                xu = eps4 / norm;
                for ( k = p; k <= q; k++ )
                {
                  rv6[k] = rv6[k] * xu;
                }
              }
    /*
      If RV1(I-1) == E(I), a row interchange was performed earlier in the
      triangularization process.
    */
              for ( i = p + 1; i <= q; i++ )
              {
                u = rv6[i];

                if ( rv1[i-1] == e[i] )
                {
                  u = rv6[i-1];
                  rv6[i-1] = rv6[i];
                }
                rv6[i] = u - rv4[i] * rv6[i-1];
              }
              its = its + 1;
            }
          }
        }

        if ( n <= q )
        {
          break;
        }
      }

      free ( rv1 );
      free ( rv2 );
      free ( rv3 );
      free ( rv4 );
      free ( rv6 );

      return ierr;
    }
    /******************************************************************************/

    int tql1 ( int n, double d[], double e[] )

    /******************************************************************************/
    /*
      Purpose:

        TQL1 computes all eigenvalues of a real symmetric tridiagonal matrix.

      Discussion:

        TQL1 finds the eigenvalues of a symmetric tridiagonal
        matrix by the QL method.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        31 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      References:

        Bowdler, Martin, Reinsch, James Wilkinson,
        Numerische Mathematik,
        Volume 11, 1968, pages 293-306.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, is the order of the matrix.

        Input/output, double D[N].
        On input, the diagonal elements of the matrix.
        On output, the eigenvalues in ascending order.
        If an error exit is made, the eigenvalues are correct and
        ordered for indices 1, 2,... IERR-1, but may not be
        the smallest eigenvalues.

        Input/output, double E[N].  On input, E(2:N) contains the
        subdiagonal elements of the input matrix, and E(1) is arbitrary.
        On output, E has been destroyed.

        Output, int TQL1, error flag.
        0, normal return,
        J, if the J-th eigenvalue has not been determined after 30 iterations.
    */
    {
      double c;
      double c2;
      double c3 = 0.0;
      double dl1;
      double el1;
      double f;
      double g;
      double h;
      int i;
      int ierr;
      //int ii;
      int j;
      int l;
      int l1;
      int l2;
      int m;
      //int mml;
      double p;
      double r;
      double s;
      double s2 = 0.0;
      double tst1;
      double tst2;

      ierr = 0;

      if ( n == 1 )
      {
        return ierr;
      }

      for ( i = 1; i < n; i++ )
      {
        e[i-1] = e[i];
      }

      f = 0.0;
      tst1 = 0.0;
      e[n-1] = 0.0;

      for ( l = 0; l < n; l++ )
      {
        j = 0;
        h = fabs ( d[l] ) + fabs ( e[l] );
        tst1 = r8_max ( tst1, h );
    /*
      Look for a small sub-diagonal element.
    */
        for ( m = l; m < n; m++ )
        {
          tst2 = tst1 + fabs ( e[m] );

          if ( tst2 == tst1 )
          {
            break;
          }
        }

        if ( m != l )
        {
          while ( true )
          {
            if ( 30 <= j )
            {
              ierr = l + 1;
              return ierr;
            }

            j = j + 1;
    /*
      Form the shift.
    */
            l1 = l + 1;
            l2 = l1 + 1;
            g = d[l];
            p = ( d[l1] - g ) / ( 2.0 * e[l] );
            r = pythag ( p, 1.0 );
            d[l] = e[l] / ( p + fabs ( r ) * r8_sign ( p ) );
            d[l1] = e[l] * ( p + fabs ( r ) * r8_sign ( p ) );
            dl1 = d[l1];
            h = g - d[l];

            for ( i = l2; i < n; i++ )
            {
              d[i] = d[i] - h;
            }

            f = f + h;
    /*
      QL transformation.
    */
            p = d[m];
            c = 1.0;
            c2 = c;
            el1 = e[l1];
            s = 0.0;

            for ( i = m - 1; l <= i; i-- )
            {
              c3 = c2;
              c2 = c;
              s2 = s;
              g = c * e[i];
              h = c * p;
              r = pythag ( p, e[i] );
              e[i+1] = s * r;
              s = e[i] / r;
              c = p / r;
              p = c * d[i] - s * g;
              d[i+1] = h + s * ( c * g + s * d[i] );
            }

            p = - s * s2 * c3 * el1 * e[l] / dl1;
            e[l] = s * p;
            d[l] = c * p;
            tst2 = tst1 + fabs ( e[l] );

            if ( tst2 <= tst1 )
            {
              break;
            }

          }
        }

        p = d[l] + f;
    /*
      Order the eigenvalues.
    */
        for ( i = l; 0 <= i; i-- )
        {
          if ( i == 0 )
          {
            d[i] = p;
          }
          else if ( d[i-1] <= p )
          {
            d[i] = p;
            break;
          }
          else
          {
            d[i] = d[i-1];
          }
        }
      }

      return ierr;
    }
    /******************************************************************************/

    int tql2 ( int n, double d[], double e[], double z[] )

    /******************************************************************************/
    /*
      Purpose:

        TQL2 computes all eigenvalues/vectors, real symmetric tridiagonal matrix.

      Discussion:

        TQL2 finds the eigenvalues and eigenvectors of a symmetric
        tridiagonal matrix by the QL method.  The eigenvectors of a full
        symmetric matrix can also be found if TRED2 has been used to reduce this
        full matrix to tridiagonal form.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Bowdler, Martin, Reinsch, Wilkinson,
        TQL2,
        Numerische Mathematik,
        Volume 11, pages 293-306, 1968.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double D[N].  On input, the diagonal elements of
        the matrix.  On output, the eigenvalues in ascending order.  If an error
        exit is made, the eigenvalues are correct but unordered for indices
        1,2,...,IERR-1.

        Input/output, double E[N].  On input, E(2:N) contains the
        subdiagonal elements of the input matrix, and E(1) is arbitrary.
        On output, E has been destroyed.

        Input, double Z[N*N].  On input, the transformation matrix
        produced in the reduction by TRED2, if performed.  If the eigenvectors of
        the tridiagonal matrix are desired, Z must contain the identity matrix.
        On output, Z contains the orthonormal eigenvectors of the symmetric
        tridiagonal (or full) matrix.  If an error exit is made, Z contains
        the eigenvectors associated with the stored eigenvalues.

        Output, int TQL2, error flag.
        0, normal return,
        J, if the J-th eigenvalue has not been determined after
        30 iterations.
    */
    {
      double c;
      double c2;
      double c3 = 0.0;
      double dl1;
      double el1;
      double f;
      double g;
      double h;
      int i;
      int ierr;
      int ii;
      int j;
      int k;
      int l;
      int l1;
      int l2;
      int m;
      int mml;
      double p;
      double r;
      double s;
      double s2 = 0.0;
      double t;
      double tst1;
      double tst2;

      ierr = 0;

      if ( n == 1 )
      {
        return ierr;
      }

      for ( i = 1; i < n; i++ )
      {
        e[i-1] = e[i];
      }

      f = 0.0;
      tst1 = 0.0;
      e[n-1] = 0.0;

      for ( l = 0; l < n; l++ )
      {
        j = 0;
        h = fabs ( d[l] ) + fabs ( e[l] );
        tst1 = r8_max ( tst1, h );
    /*
      Look for a small sub-diagonal element.
    */
        for ( m = l; m < n; m++ )
        {
          tst2 = tst1 + fabs ( e[m] );
          if ( tst2 == tst1 )
          {
            break;
          }
        }

        if ( m != l )
        {
          for ( ; ; )
          {
            if ( 30 <= j )
            {
              ierr = l + 1;
              return ierr;
            }

            j = j + 1;
    /*
      Form shift.
    */
            l1 = l + 1;
            l2 = l1 + 1;
            g = d[l];
            p = ( d[l1] - g ) / ( 2.0 * e[l] );
            r = pythag ( p, 1.0 );
            d[l] = e[l] / ( p + r8_sign ( p ) * fabs ( r ) );
            d[l1] = e[l] * ( p + r8_sign ( p ) * fabs ( r ) );
            dl1 = d[l1];
            h = g - d[l];
            for ( i = l2; i < n; i++ )
            {
              d[i] = d[i] - h;
            }
            f = f + h;
    /*
      QL transformation.
    */
            p = d[m];
            c = 1.0;
            c2 = c;
            el1 = e[l1];
            s = 0.0;
            mml = m - l;

            for ( ii = 1; ii <= mml; ii++ )
            {
              c3 = c2;
              c2 = c;
              s2 = s;
              i = m - ii;
              g = c * e[i];
              h = c * p;
              r = pythag ( p, e[i] );
              e[i+1] = s * r;
              s = e[i] / r;
              c = p / r;
              p = c * d[i] - s * g;
              d[i+1] = h + s * ( c * g + s * d[i] );
    /*
      Form vector.
    */
              for ( k = 0; k < n; k++ )
              {
                h = z[k+(i+1)*n];
                z[k+(i+1)*n] = s * z[k+i*n] + c * h;
                z[k+i*n] = c * z[k+i*n] - s * h;
              }
            }
            p = - s * s2 * c3 * el1 * e[l] / dl1;
            e[l] = s * p;
            d[l] = c * p;
            tst2 = tst1 + fabs ( e[l] );

            if ( tst2 <= tst1 )
            {
              break;
            }
          }
        }
        d[l] = d[l] + f;
      }
    /*
      Order eigenvalues and eigenvectors.
    */
      for ( ii = 1; ii < n; ii++ )
      {
        i = ii - 1;
        k = i;
        p = d[i];
        for ( j = ii; j < n; j++ )
        {
          if ( d[j] < p )
          {
            k = j;
            p = d[j];
          }
        }

        if ( k != i )
        {
          d[k] = d[i];
          d[i] = p;
          for ( j = 0; j < n; j++ )
          {
            t        = z[j+i*n];
            z[j+i*n] = z[j+k*n];
            z[j+k*n] = t;
          }
        }
      }
      return ierr;
    }
    /******************************************************************************/

    int tqlrat ( int n, double d[], double e2[] )

    /******************************************************************************/
    /*
      Purpose:

        TQLRAT computes all eigenvalues of a real symmetric tridiagonal matrix.

      Discussion:

        TQLRAT finds the eigenvalues of a symmetric
        tridiagonal matrix by the rational QL method.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Christian Reinsch,
        Algorithm 464, TQLRAT,
        Communications of the ACM,
        Volume 16, page 689, 1973.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input/output, double D[N].  On input, D contains the diagonal
        elements of the matrix.  On output, D contains the eigenvalues in ascending
        order.  If an error exit was made, then the eigenvalues are correct
        in positions 1 through IERR-1, but may not be the smallest eigenvalues.

        Input/output, double E2[N], contains in positions 2 through N
        the squares of the subdiagonal elements of the matrix.  E2(1) is
        arbitrary.  On output, E2 has been overwritten by workspace
        information.

        Output, int TQLRAT, error flag.
        0, for no error,
        J, if the J-th eigenvalue could not be determined after 30 iterations.
    */
    {
      double b = 0.0;
      double c = 0.0;
      double f;
      double g;
      double h;
      int i;
      int ierr;
      int ii;
      int j;
      int l;
      int l1;
      int m;
      int mml;
      double p;
      double r;
      double s;
      double t;

      ierr = 0;

      if ( n == 1 )
      {
        return ierr;
      }

      for ( i = 1; i < n; i++ )
      {
        e2[i-1] = e2[i];
      }

      f = 0.0;
      t = 0.0;
      e2[n-1] = 0.0;

      for ( l = 0; l < n; l++ )
      {
         j = 0;
         h = fabs ( d[l] ) + sqrt ( e2[l] );

         if ( t <= h )
         {
           t = h;
           b = fabs ( t ) * r8_epsilon ( );
           c = b * b;
         }
    /*
      Look for small squared sub-diagonal element.
    */
        for ( m = l; m < n; m++ )
        {
          if ( e2[m] <= c )
          {
            break;
          }
        }

        if ( m != l )
        {
          for ( ; ; )
          {
            if ( 30 <= j )
            {
              ierr = l + 1;
              return ierr;
            }

            j = j + 1;
    /*
      Form shift.
    */
            l1 = l + 1;
            s = sqrt ( e2[l] );
            g = d[l];
            p = ( d[l1] - g ) / ( 2.0 * s );
            r = pythag ( p, 1.0 );
            d[l] = s / ( p + fabs ( r ) * r8_sign ( p ) );
            h = g - d[l];
            for ( i = l1; i < n; i++ )
            {
              d[i] = d[i] - h;
            }
            f = f + h;
    /*
      Rational QL transformation.
    */
            g = d[m];
            if ( g == 0.0 )
            {
              g = b;
            }

            h = g;
            s = 0.0;
            mml = m - l;

            for ( ii = 1; ii <= mml; ii++ )
            {
              i = m - ii;
              p = g * h;
              r = p + e2[i];
              e2[i+1] = s * r;
              s = e2[i] / r;
              d[i+1] = h + s * ( h + d[i] );
              g = d[i] - e2[i] / g;
              if ( g == 0.0 )
              {
                g = b;
              }
              h = g * p / r;
            }
            e2[l] = s * g;
            d[l] = h;
    /*
      Guard against underflow in convergence test.
    */
            if ( h == 0.0 )
            {
              break;
            }

            if ( fabs ( e2[l] ) <= fabs ( c / h ) )
            {
              break;
            }

            e2[l] = h * e2[l];

            if ( e2[l] == 0.0 )
            {
              break;
            }
          }
        }

        p = d[l] + f;
    /*
      Order the eigenvalues.
    */
        for ( i = l; 0 <= i; i-- )
        {
          if ( i == 0 )
          {
            d[i] = p;
            break;
          }
          else if ( d[i-1] <= p )
          {
            d[i] = p;
            break;
          }
          d[i] = d[i-1];
        }
      }

      return ierr;
    }
    /******************************************************************************/

    void trbak1 ( int n, double a[], double e[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        TRBAK1 determines eigenvectors by undoing the TRED1 transformation.

      Discussion:

        TRBAK1 forms the eigenvectors of a real symmetric
        matrix by back transforming those of the corresponding
        symmetric tridiagonal matrix determined by TRED1.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        27 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double A[N,N], contains information about the orthogonal
        transformations used in the reduction by TRED1 in its strict lower
        triangle.

        Input, double E[N], the subdiagonal elements of the tridiagonal
        matrix in E(2:N).  E(1) is arbitrary.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double Z[N,M].  On input, the eigenvectors to be
        back transformed.  On output, the transformed eigenvectors.
    */
    {
      int i;
      int j;
      int k;
      int l;
      double s;

      if ( m <= 0 )
      {
        return;
      }

      if ( n <= 1 )
      {
        return;
      }

      for ( i = 1; i < n; i++ )
      {
        l = i - 1;

        if ( e[i] != 0.0 )
        {
          for ( j = 0; j < m; j++ )
          {
            s = 0.0;
            for ( k = 0; k < i; k++ )
            {
              s = s + a[i+k*n] * z[k+j*n];
            }
            s = ( s / a[i+l*n] ) / e[i];
            for ( k = 0; k < i; k++ )
            {
              z[k+j*n] = z[k+j*n] + s * a[i+k*n];
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void trbak3 ( int n, int nv, double a[], int m, double z[] )

    /******************************************************************************/
    /*
      Purpose:

        TRBAK3 determines eigenvectors by undoing the TRED3 transformation.

      Discussion:

        TRBAK3 forms the eigenvectors of a real symmetric
        matrix by back transforming those of the corresponding
        symmetric tridiagonal matrix determined by TRED3.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        28 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

       Reference:

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int NV, the dimension of the array paramater A,
        which must be at least N*(N+1)/2.

        Input, double A[NV], information about the orthogonal
        transformations used in the reduction by TRED3.

        Input, int M, the number of eigenvectors to be back
        transformed.

        Input/output, double Z[N,M].  On input, the eigenvectors to be
        back transformed.  On output, the transformed eigenvectors.
    */
    {
      nv  = 1000;
      double h;
      int i;
      int ik;
      int iz;
      int j;
      int k;
      double s;

      if ( m == 0 )
      {
        return;
      }

      for ( i = 1; i < n; i++ )
      {
        iz = ( i * ( i + 1 ) ) / 2;
        ik = iz + i;
        h = a[ik];

        if ( h != 0.0 )
        {
          for ( j = 0; j < m; j++ )
          {
            s = 0.0;
            ik = iz - 1;

            for ( k = 0; k < i; k++ )
            {
              ik = ik + 1;
              s = s + a[ik] * z[k+j*n];
            }

            s = ( s / h ) / h;
            ik = iz - 1;

            for ( k = 0; k < i; k++ )
            {
              ik = ik + 1;
              z[k+j*n] = z[k+j*n] - s * a[ik];
            }
          }
        }
      }

      return;
    }
    /******************************************************************************/

    void tred1 ( int n, double a[], double d[], double e[], double e2[] )

    /******************************************************************************/
    /*
      Purpose:

        TRED1 transforms a real symmetric matrix to symmetric tridiagonal form.

      Discussion:

        TRED1 reduces a real symmetric matrix to a symmetric
        tridiagonal matrix using orthogonal similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        08 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Martin, Reinsch, Wilkinson,
        TRED1,
        Numerische Mathematik,
        Volume 11, pages 181-195, 1968.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix A.

        Input/output, double A[N*N], on input, contains the real
        symmetric matrix.  Only the lower triangle of the matrix need be supplied.
        On output, A contains information about the orthogonal transformations
        used in the reduction in its strict lower triangle.
        The full upper triangle of A is unaltered.

        Output, double D[N], contains the diagonal elements of the
        tridiagonal matrix.

        Output, double E[N], contains the subdiagonal elements of the
        tridiagonal matrix in its last N-1 positions.  E(1) is set to zero.

        Output, double E2[N], contains the squares of the corresponding
        elements of E.  E2 may coincide with E if the squares are not needed.
    */
    {
      double f;
      double g;
      double h;
      int i;
      //int ii;
      int j;
      int k;
      int l;
      double scale;

      for ( j = 0; j < n; j++ )
      {
        d[j] = a[n-1+j*n];
      }

      for ( i = 0; i < n; i++ )
      {
        a[n-1+i*n] = a[i+i*n];
      }

      for ( i = n - 1; 0 <= i; i-- )
      {
        l = i - 1;
        h = 0.0;
    /*
      Scale row.
    */
        scale = 0.0;
        for ( k = 0; k <= l; k++ )
        {
          scale = scale + fabs ( d[k] );
        }

        if ( scale == 0.0 )
        {
          for ( j = 0; j <= l; j++ )
          {
            d[j]     = a[l+j*n];
            a[l+j*n] = a[i+j*n];
            a[i+j*n] = 0.0;
          }

          e[i] = 0.0;
          e2[i] = 0.0;
          continue;
        }

        for ( k = 0; k <= l; k++ )
        {
          d[k] = d[k] / scale;
        }

        for ( k = 0; k <= l; k++ )
        {
          h = h + d[k] * d[k];
        }

        e2[i] = h * scale * scale;
        f = d[l];
        g = - sqrt ( h ) * r8_sign ( f );
        e[i] = scale * g;
        h = h - f * g;
        d[l] = f - g;

        if ( 0 <= l )
        {
    /*
      Form A * U.
    */
          for ( k = 0; k <= l; k++ )
          {
            e[k] = 0.0;
          }

          for ( j = 0; j <= l; j++ )
          {
            f = d[j];
            g = e[j] + a[j+j*n] * f;

            for ( k = j + 1; k <= l; k++ )
            {
              g = g + a[k+j*n] * d[k];
              e[k] = e[k] + a[k+j*n] * f;
            }
            e[j] = g;
          }
    /*
      Form P.
    */
          f = 0.0;
          for ( j = 0; j <= l; j++ )
          {
            e[j] = e[j] / h;
            f = f + e[j] * d[j];
          }

          h = f / ( h + h );
    /*
      Form Q.
    */
          for ( j = 0; j <= l; j++ )
          {
            e[j] = e[j] - h * d[j];
          }
    /*
      Form reduced A.
    */
          for ( j = 0; j <= l; j++ )
          {
            f = d[j];
            g = e[j];
            for ( k = j; k <= l; k++ )
            {
              a[k+j*n] = a[k+j*n] - f * e[k] - g * d[k];
            }
          }
        }

        for ( j = 0; j <= l; j++ )
        {
          f        = d[j];
          d[j]     = a[l+j*n];
          a[l+j*n] = a[i+j*n];
          a[i+j*n] = f * scale;
        }
      }
      return;
    }
    /******************************************************************************/

    void tred2 ( int n, double a[], double d[], double e[], double z[] )

    /******************************************************************************/
    /*
      Purpose:

        TRED2 transforms a real symmetric matrix to symmetric tridiagonal form.

      Discussion:

        TRED2 reduces a real symmetric matrix to a
        symmetric tridiagonal matrix using and accumulating
        orthogonal similarity transformations.

        A and Z may coincide, in which case a single storage area is used
        for the input of A and the output of Z.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        03 November 2012

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Martin, Reinsch, Wilkinson,
        TRED2,
        Numerische Mathematik,
        Volume 11, pages 181-195, 1968.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, double A[N*N], the real symmetric input matrix.  Only the
        lower triangle of the matrix need be supplied.

        Output, double D[N], the diagonal elements of the tridiagonal
        matrix.

        Output, double E[N], contains the subdiagonal elements of the
        tridiagonal matrix in E(2:N).  E(1) is set to zero.

        Output, double Z[N*N], the orthogonal transformation matrix
        produced in the reduction.
    */
    {
      double f;
      double g;
      double h;
      double hh;
      int i;
      //int ii;
      int j;
      int k;
      int l;
      double scale;

      for ( j = 0; j < n; j++ )
      {
        for ( i = j; i < n; i++ )
        {
          z[i+j*n] = a[i+j*n];
        }
      }

      for ( j = 0; j < n; j++ )
      {
        d[j] = a[n-1+j*n];
      }

      for ( i = n - 1; 1 <= i; i-- )
      {
        l = i - 1;
        h = 0.0;
    /*
      Scale row.
    */
        scale = 0.0;
        for ( k = 0; k <= l; k++ )
        {
          scale = scale + fabs ( d[k] );
        }

        if ( scale == 0.0 )
        {
          e[i] = d[l];

          for ( j = 0; j <= l; j++ )
          {
            d[j]     = z[l+j*n];
            z[i+j*n] = 0.0;
            z[j+i*n] = 0.0;
          }
          d[i] = 0.0;
          continue;
        }

        for ( k = 0; k <= l; k++ )
        {
          d[k] = d[k] / scale;
        }

        h = 0.0;
        for ( k = 0; k <= l; k++ )
        {
          h = h + d[k] * d[k];
        }

        f = d[l];
        g = - sqrt ( h ) * r8_sign ( f );
        e[i] = scale * g;
        h = h - f * g;
        d[l] = f - g;
    /*
      Form A*U.
    */
        for ( k = 0; k <= l; k++ )
        {
          e[k] = 0.0;
        }

        for ( j = 0; j <= l; j++ )
        {
          f = d[j];
          z[j+i*n] = f;
          g = e[j] + z[j+j*n] * f;

          for ( k = j + 1; k <= l; k++ )
          {
            g = g + z[k+j*n] * d[k];
            e[k] = e[k] + z[k+j*n] * f;
          }
          e[j] = g;
        }
    /*
      Form P.
    */
        for ( k = 0; k <= l; k++ )
        {
          e[k] = e[k] / h;
        }
        f = 0.0;
        for ( k = 0; k <= l; k++ )
        {
          f = f + e[k] * d[k];
        }
        hh = 0.5 * f / h;
    /*
      Form Q.
    */
        for ( k = 0; k <= l; k++ )
        {
          e[k] = e[k] - hh * d[k];
        }
    /*
      Form reduced A.
    */
        for ( j = 0; j <= l; j++ )
        {
          f = d[j];
          g = e[j];

          for ( k = j; k <= l; k++ )
          {
            z[k+j*n] = z[k+j*n] - f * e[k] - g * d[k];
          }
          d[j] = z[l+j*n];
          z[i+j*n] = 0.0;
        }
        d[i] = h;
      }
    /*
      Accumulation of transformation matrices.
    */
      for ( i = 1; i < n; i++ )
      {
        l = i - 1;
        z[n-1+l*n] = z[l+l*n];
        z[l+l*n] = 1.0;
        h = d[i];

        if ( h != 0.0 )
        {
          for ( k = 0; k <= l; k++ )
          {
            d[k] = z[k+i*n] / h;
          }
          for ( j = 0; j <= l; j++ )
          {
            g = 0.0;
            for ( k = 0; k <= l; k++ )
            {
              g = g + z[k+i*n] * z[k+j*n];
            }
            for ( k = 0; k <= l; k++ )
            {
              z[k+j*n] = z[k+j*n] - g * d[k];
            }
          }
        }
        for ( k = 0; k <= l; k++ )
        {
          z[k+i*n] = 0.0;
        }
      }

      for ( j = 0; j < n; j++ )
      {
        d[j] = z[n-1+j*n];
      }

      for ( j = 0; j < n - 1; j++ )
      {
        z[n-1+j*n] = 0.0;
      }
      z[n-1+(n-1)*n] = 1.0;

      e[0] = 0.0;

      return;
    }
    /******************************************************************************/

    void tred3 ( int n, int nv, double a[], double d[], double e[], double e2[] )

    /******************************************************************************/
    /*
      Purpose:

        TRED3: transform real symmetric packed matrix to symmetric tridiagonal form.

      Discussion:

        TRED3 reduces a real symmetric matrix, stored as
        a one-dimensional array, to a symmetric tridiagonal matrix
        using orthogonal similarity transformations.

      Licensing:

        This code is distributed under the GNU LGPL license.

      Modified:

        27 January 2018

      Author:

        Original FORTRAN77 version by Smith, Boyle, Dongarra, Garbow, Ikebe,
        Klema, Moler.
        C version by John Burkardt.

      Reference:

        Martin, Reinsch, James Wilkinson,
        TRED3,
        Numerische Mathematik,
        Volume 11, pages 181-195, 1968.

        James Wilkinson, Christian Reinsch,
        Handbook for Automatic Computation,
        Volume II, Linear Algebra, Part 2,
        Springer, 1971,
        ISBN: 0387054146,
        LC: QA251.W67.

        Brian Smith, James Boyle, Jack Dongarra, Burton Garbow,
        Yasuhiko Ikebe, Virginia Klema, Cleve Moler,
        Matrix Eigensystem Routines, EISPACK Guide,
        Lecture Notes in Computer Science, Volume 6,
        Springer Verlag, 1976,
        ISBN13: 978-3540075462,
        LC: QA193.M37.

      Parameters:

        Input, int N, the order of the matrix.

        Input, int NV, the dimension of A, which must be at least
        (N*(N+1))/2.

        Input/output, double A[NV].  On input, the lower triangle of
        the real symmetric matrix, stored row-wise.  On output, information about
        the orthogonal transformations used in the reduction.

        Output, double D[N], the diagonal elements of the tridiagonal
        matrix.

        Output, double E[N], the subdiagonal elements of the tridiagonal
        matrix in E(2:N).  E(1) is set to zero.

        Output, double E2[N],  the squares of the corresponding
        elements of E.  E2 may coincide with E if the squares are not needed.
    */
    {
      nv = 1000;
      double f;
      double g;
      double h;
      double hh;
      int i;
      int iz;
      int j;
      int jk;
      int k;
      double scale;

      for ( i = n - 1; 0 <= i; i-- )
      {
        iz = ( i * ( i + 1 ) ) / 2 - 1;
        h = 0.0;
        scale = 0.0;
    /*
      Scale row.
    */
        for ( k = 0; k < i; k++ )
        {
          iz = iz + 1;
          d[k] = a[iz];
          scale = scale + fabs ( d[k] );
        }

        if ( scale == 0.0 )
        {
          e[i] = 0.0;
          e2[i] = 0.0;
          d[i] = a[iz+1];
          a[iz+1] = scale * sqrt ( h );
          continue;
        }

        for ( k = 0; k < i; k++ )
        {
          d[k] = d[k] / scale;
          h = h + d[k] * d[k];
        }

        e2[i] = scale * scale * h;
        f = d[i-1];
        g = - sqrt ( h ) * r8_sign ( f );
        e[i] = scale * g;
        h = h - f * g;
        d[i-1] = f - g;
        a[iz] = scale * d[i-1];
        if ( i == 1 )
        {
          d[i] = a[iz+1];
          a[iz+1] = scale * sqrt ( h );
          continue;
        }

        jk = 0;

        for ( j = 0; j < i; j++ )
        {
          f = d[j];
          g = 0.0;

          for ( k = 0; k < j; k++ )
          {
            g = g + a[jk] * d[k];
            e[k] = e[k] + a[jk] * f;
            jk = jk + 1;
          }

          e[j] = g + a[jk] * f;
          jk = jk + 1;
        }
    /*
      Form P.
    */
        for ( j = 0; j < i; j++ )
        {
          e[j] = e[j] / h;
        }
        f = 0.0;
        for ( j = 0; j < i; j++ )
        {
          f = f + e[j] * d[j];
        }
        hh = f / ( h + h );
    /*
      Form Q.
    */
        for ( j = 0; j < i; j++ )
        {
          e[j] = e[j] - hh * d[j];
        }
        jk = 0;
    /*
      Form reduced A.
    */
        for ( j = 0; j < i; j++ )
        {
          f = d[j];
          g = e[j];
          for ( k = 0; k <= j; k++ )
          {
            a[jk] = a[jk] - f * e[k] - g * d[k];
            jk = jk + 1;
          }
        }

        d[i] = a[iz+1];
        a[iz+1] = scale * sqrt ( h );
      }

      return;
    }

}
