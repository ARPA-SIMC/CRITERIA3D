/*!
     \brief Kriging Interpolator

     based on: Chao-yi Lang
     July, 1995
     lang@cs.cornell.edu
*/

    #include <stdlib.h>
    #include <math.h>
    #include <stdio.h>

    #include "kriging.h"


    /*! global variables */
    static short mode;
    static int dim;
    static double range, nugget, sill, sill_nugget, slope;
    static double *D, *V, *weight, *VM, *pos, *val;


    /*!
     * \brief no pivoting - assume rank n
     * \param A matrix double pointer
     * \return true on success, false otherwise
     */
    bool matrixInversion(double *A)
    {
        double *inverseA, rapporto;
        int i, j, iter, first_row;

        inverseA = (double *) malloc(sizeof(double) * dim *dim);

        /*! inizializza = matrice Identita' */
        for (i = 0; i < dim; i++)
            for(j = 0; j < dim; j++)
                inverseA[i*dim+j] = (i == j) ? 1. : 0. ;

        /*! step 1: A diventa triangolare sup , inverseA triangolare inf */
        for (iter = 0; iter < (dim-1); iter++)
        {
            first_row = iter + 1;
            for(i = first_row; i < dim; i++)
            {
                rapporto = A[i*dim+iter] / A[iter*dim+iter];
                A[i*dim+iter] = 0.;
                for(j = first_row; j < dim; j++)
                    A[i*dim+j] -= rapporto * A[iter*dim+j];
                for(j = 0; j <= iter; j++)
                    inverseA[i*dim+j] -= rapporto * inverseA[iter*dim+j];
            }
        }

        /*! step 2: A diventa diagonale */
        for (iter = dim - 1; iter > 0; iter--)
        {
            first_row = iter - 1;
            for(i = first_row; i >= 0; i--)
            {
                rapporto = A[i*dim+iter] / A[iter*dim+iter];
                A[i*dim+iter] = 0.;
                for(j = 0; j < dim; j++)
                    inverseA[i*dim+j] -= rapporto * inverseA[iter*dim+j];
            }
        }

        /*! step 3: divide per la diagonale e ottiene inverseA */
        for(i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                inverseA[i*dim+j] /= A[i*dim+i];

        /*! salva matrice inversa e libera memoria */
        for(i = 0; i < dim; i++)
            for (j = 0; j < dim; j++)
                A[i*dim+j] = inverseA[i*dim+j];

        free(inverseA);

        return true;
    }



    /*!
     * \brief costruisce la matrice delle distanze e il variogramma
     * \param myPos pos[N*2]	[i*2]   x i-esima stazione  [i*2+1] y i-esima stazione
     * \param myVal dati stazione
     * \param nrItems numero di stazioni
     * \param myMode mode 1-Spher mode 2-Expon mode 3-Gauss mode 4-Linear mode
     * \param myRange
     * \param myNugget
     * \param mySill
     * \param mySlope
     * \return
     */
    bool krigingVariogram(double *myPos, double *myVal, int nrItems, short myMode,
                          double myRange, double myNugget, double mySill, double mySlope)
    {
        int i, j;
        double dx, dy, tmp;
        double *Cd;

        /*! global variables */
        mode	= myMode;
        dim		= nrItems + 1;
        range = myRange;
        nugget = myNugget;
        sill = mySill;
        slope = mySlope;
        sill_nugget = sill - nugget;

        krigingFreeMemory();

        V		= (double *) malloc (sizeof (double) * dim * dim );
        Cd		= (double *) malloc (sizeof (double) * dim * dim );
        D		= (double *) malloc (sizeof (double) * dim);
        weight	= (double *) malloc (sizeof (double) * dim);
        pos		= (double *) malloc (sizeof (double) * nrItems * 2);
        val		= (double *) malloc (sizeof (double) * nrItems);

        /*! salva matrici posizione e valore */
        for (i=0; i < nrItems; i++)
        {
            pos[i*2]	= myPos[i*2];
            pos[i*2+1]	= myPos[i*2+1];
            val[i]		= myVal[i];
        }

        /*! matrice (triangolare sup) delle distanze */
        for (i=0; i < nrItems; i++)
        {
            for (j=i; j < nrItems; j++)
            {
                dx = pos[i*2]-pos[j*2];
                dy = pos[i*2+1]-pos[j*2+1];
                Cd[(i*dim)+j] = sqrt(dx * dx + dy * dy);
            }
        }

        /*! inizializza ultima riga e colonna del variogramma
        * es:	0 0 0 1
        *		0 0 0 1
        *		0 0 0 1
        *		1 1 1 0 */

        for (i = 0; i < nrItems; i++)
        {
            V[i*dim +dim -1] = 1;
            V[(dim-1)*dim +i] = 1;
        }
        V[(dim-1)*(dim)+i] = 0;

        /*! calcola il variogramma e lo salva in V */
        for (i = 0; i < nrItems; i++)
        {
            for (j = i; j < nrItems; j++)
            {
                tmp = Cd[i*dim+j] / range;
                switch (mode)
                {
                /*! Spherical */
                    case 1 :
                            if (Cd[i*dim+j] < range)
                            {
                                V[i*dim+j] = V[j*dim+i] = nugget + sill_nugget *
                                            (1.5 * tmp - 0.5 * tmp * tmp * tmp);
                            }
                            else V[i*dim+j] = V[j*dim+i] = nugget + sill_nugget;
                            break;
                /*! Exponential */
                    case 2 :
                            V[i*dim+j] = V[j*dim+i] = nugget + sill_nugget *
                                        (1. - exp(-3.* tmp));
                            break;

                /*! Gaussian */
                    case 3 :
                            V[i*dim+j] = V[j*dim+i] = nugget + sill_nugget *
                                        (1. - exp(-4.* tmp * tmp));
                            break;

                /*! Linear */
                    case 4 :
                            V[i*dim+j] = V[j*dim+i] = nugget + slope * Cd[i*dim+j];
                            break;

                /*! Default: linear */
                    default:
                            V[i*dim+j] = V[j*dim+i] = nugget + slope * Cd[i*dim+j];
                            break;
                }
            }
        }

        /*! inverte variogramma e libera memoria*/
        matrixInversion(V);

        free(Cd);

        return true;
    }


/*!
 * \brief assegna i pesi dei punti misura rispetto a P(x,y) e li salva nella matrice weight
 * \param x_p x point
 * \param y_p y point
 * \return true on success, false otherwise
 */
bool krigingSetWeight(double x_p, double y_p)
    {
        int i, j;
        double dx, dy, tmp, h;

        /*! calcola le distanze tra P e i punti di misura e calcola il variogramma (memorizzato in D) */
        for (i=0; i < dim-1; i++)
            {
            dx = pos[i*2] - x_p;
            dy = pos[i*2+1] - y_p;
            h = sqrt(dx * dx + dy * dy);
            tmp = h / range;
            switch( mode )
                {
            /*! Sferico */
                case 1 :
                         if ( h < range )
                            D[i] = nugget + sill_nugget * (1.5 * tmp - 0.5 * tmp * tmp * tmp);
                         else
                            D[i] = nugget + sill_nugget;
                         break;
            /*! Esponenziale */
                case 2 :
                         D[i] = nugget + sill_nugget * (1. - exp(-3.* tmp));
                         break;

            /*! Gaussiano */
                case 3 :
                         D[i] = nugget + sill_nugget * (1. - exp(-4.* tmp * tmp));
                         break;

            /*! Lineare */
                case 4 :
                         D[i] = nugget + slope * h;
                         break;

            /*! Default: Lineare */
                default:
                         D[i] = nugget + slope * h;
                         break;
                }
        }
        D[dim-1] = 1;

        /*! calcola i pesi */
        for (i=0; i < dim-1; i++)
            {
            weight[i] = 0;
            for (j=0; j < dim; j++)
                weight[i] += V[i*dim+j] * D[j];
            }

        return true;
    }


/*!
 * \brief return interpolated value in the point (x_p, y_p)
 * \return result
 */
double krigingResult()
    {
        double result = 0.0;

        for (int i=0; i < dim-1; i++)
            result += weight[i] * val[i];

        return result;
    }



/*!
 * \brief Root Mean Square Error
 * \return result
 */
double krigingRMSE()
    {
        double error = 0.0;

        for (int i=0; i < dim-1; i++)
            error += weight[i] * D[i];

        return sqrt(fabs(error));
    }



void krigingFreeMemory()
    {
        if (V != nullptr)
        {
            free(V);
            V = nullptr;
        }
        if (VM != nullptr)
        {
            free(VM);
            VM = nullptr;
        }
        if (pos != nullptr)
        {
            free(pos);
            pos = nullptr;
        }
        if (val != nullptr)
        {
            free(val);
            val = nullptr;
        }
        if (D != nullptr)
        {
            free(D);
            D = nullptr;
        }
        if (weight != nullptr)
        {
            free(weight);
            weight = nullptr;
        }
    }
