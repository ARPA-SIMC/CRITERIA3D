#include <math.h>
#include <float.h>
#include <limits.h>             // required for LONG_MAX
#include "commonConstants.h"
#include "gammaFunction.h"
#include "furtherMathFunctions.h"


    static long double xSmall_x(long double x, long double nu);
    static long double xMedium_x(long double x, long double nu);
    static long double xLarge_x(long double x, long double nu);
    static long double xGamma(long double x);
    static long double Duplication_Formula( long double two_x );
    static long double xLnGamma_Asymptotic_Expansion( long double x );


    ////////////////////////////////////////////////////////////////////////////////
    // double Entire_Incomplete_Gamma_Function(double x, double nu)               //
    //                                                                            //
    //  Description:                                                              //
    //     The entire incomplete gamma function, also called the regularized      //
    //     incomplete gamma function, is defined as the integral from 0 to x of   //
    //     the integrand t^(nu-1) exp(-t) / gamma(nu) dt.  The parameter nu is    //
    //     sometimes referred to as the shape parameter.                          //
    //                                                                            //
    //  Arguments:                                                                //
    //     double x   Upper limit of the integral with integrand given above.     //
    //     double nu  The shape parameter of the entire incomplete gamma function.//
    //                                                                            //
    //  Return Values:                                                            //
    //                                                                            //
    //  Example:                                                                  //
    //     double x, g, nu;                                                       //
    //                                                                            //
    //     g = Entire_Incomplete_Gamma_Function( x, nu );                         //
    ////////////////////////////////////////////////////////////////////////////////
    double Entire_Incomplete_Gamma_Function(double x, double nu)
    {
       return (double) xEntire_Incomplete_Gamma_Function((long double)x,
                                                                  (long double)nu);
    }


    ////////////////////////////////////////////////////////////////////////////////
    // long double xEntire_Incomplete_Gamma_Function(long double x,               //
    //                                                            long double nu) //
    //                                                                            //
    //  Description:                                                              //
    //     The entire incomplete gamma function, also called the regularized      //
    //     incomplete gamma function, is defined as the integral from 0 to x of   //
    //     the integrand t^(nu-1) exp(-t) / gamma(nu) dt.  The parameter nu is    //
    //     sometimes referred to as the shape parameter.                          //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Upper limit of the integral with integrand given above.//
    //     long double nu  The shape parameter of the entire incomplete gamma     //
    //                     function.                                              //
    //                                                                            //
    //  Return Values:                                                            //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x, g, nu;                                                  //
    //                                                                            //
    //     g = xEntire_Incomplete_Gamma_Function( x, nu );                        //
    ////////////////////////////////////////////////////////////////////////////////
    long double xEntire_Incomplete_Gamma_Function(long double x, long double nu)
    {

       if (x == 0.0L) return 0.0L;
       if (fabsl(x) <= 1.0L) return xSmall_x(x, nu);
       if (fabsl(x) < (nu + 1.0L) ) return xMedium_x(x, nu);
       return xLarge_x(x, nu);
    }


    ////////////////////////////////////////////////////////////////////////////////
    // static long double xSmall_x(long double x, long double nu)                 //
    //                                                                            //
    //  Description:                                                              //
    //     This function approximates the entire incomplete gamma function for    //
    //     x, where -1 <= x <= 1.                                                 //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Upper limit of the integral with integrand described   //
    //                     in the section under Entire_Incomplete_Gamma_Function. //
    //     long double nu  The shape parameter of the entire incomplete gamma     //
    //                     function.                                              //
    //                                                                            //
    //  Return Values:                                                            //
    //     The entire incomplete gamma function:                                  //
    //                  I(0,x) t^(nu-1) Exp(-t) dt / Gamma(nu).                   //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x, g, nu;                                                  //
    //                                                                            //
    //     g = xSmall_x( x, nu);                                                  //
    ////////////////////////////////////////////////////////////////////////////////

    static long double xSmall_x(long double x, long double nu)
    {
       #define Nterms 20

       long double terms[Nterms];
       long double x_term = -x;
       long double x_power = 1.0L;
       long double sum;
       int i;

       for (i = 0; i < Nterms; i++) {
          terms[i] = (x_power / xFactorial(i)) / (i + nu);
          x_power *= x_term;
       }
       sum = terms[Nterms-1];
       for (i = Nterms-2; i >= 0; i--) sum += terms[i];
       if ( nu <= Gamma_Function_Max_Arg() )
          return powl(x,nu) * sum / xGamma_Function(nu);
       else return expl(nu * logl(x) + logl(sum) - xLn_Gamma_Function(nu));
    }


    ////////////////////////////////////////////////////////////////////////////////
    // static long double xMedium_x(long double x, long double nu)                //
    //                                                                            //
    //  Description:                                                              //
    //     This function approximates the entire incomplete gamma function for    //
    //     x, where 1 < x < nu + 1.                                               //
    //                                                                            //
    //     If nu + 1 < x, then one should use xLarge_x(x,nu).                     //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Upper limit of the integral with integrand described   //
    //                     in the section under Entire_Incomplete_Gamma_Function. //
    //     long double nu  The shape parameter of the entire incomplete gamma     //
    //                     function.                                              //
    //                                                                            //
    //  Return Values:                                                            //
    //     The entire incomplete gamma function:                                  //
    //                  I(0,x) t^(nu-1) exp(-t) dt / gamma(nu).                   //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x, g, nu;                                                  //
    //                                                                            //
    //     g = xMedium_x( x, nu);                                                 //
    ////////////////////////////////////////////////////////////////////////////////
    static long double xMedium_x(long double x, long double nu)
    {
       long double coef;
       long double term = 1.0L / nu;
       long double corrected_term = term;
       long double temp_sum = term;
       long double correction = -temp_sum + corrected_term;
       long double sum1 = temp_sum;
       long double sum2;
       long double epsilon = 0.0L;
       int i;

       if (nu > Gamma_Function_Max_Arg()) {
          coef = expl( nu * logl(x) - x - xLn_Gamma_Function(nu) );
          if (coef > 0.0L) epsilon = DBL_EPSILON/coef;
       } else {
          coef = powl(x, nu) * expl(-x) / xGamma_Function(nu);
          epsilon = DBL_EPSILON/coef;
       }
       if (epsilon <= 0.0L) epsilon = (long double) DBL_EPSILON;

       for (i = 1; term > epsilon * sum1; i++) {
          term *= x / (nu + i);
          corrected_term = term + correction;
          temp_sum = sum1 + corrected_term;
          correction = (sum1 - temp_sum) + corrected_term;
          sum1 = temp_sum;
       }
       sum2 = sum1;
       sum1 *= coef;
       correction += sum2 - sum1 / coef;
       term *= x / (nu + i);
       sum2 = term + correction;
       for (i++; (term + correction) > epsilon * sum2; i++) {
          term *= x / (nu + i);
          corrected_term = term + correction;
          temp_sum = sum2 + corrected_term;
          correction = (sum2 - temp_sum) + corrected_term;
          sum2 = temp_sum;
       }
       sum2 += correction;
       sum2 *= coef;
       return sum1 + sum2;
    }


    ////////////////////////////////////////////////////////////////////////////////
    // static long double xLarge_x(long double x, long double nu)                 //
    //                                                                            //
    //  Description:                                                              //
    //     This function approximates the entire incomplete gamma function for    //
    //     x, where nu + 1 <= x.                                                  //
    //                                                                            //
    //     If 0 <= x < nu + 1, then one should use xSmall_x(x,nu).                //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Upper limit of the integral with integrand described   //
    //                     in the section under Entire_Incomplete_Gamma_Function. //
    //     long double nu  The shape parameter of the entire incomplete gamma     //
    //                     function.                                              //
    //                                                                            //
    //  Return Values:                                                            //
    //     If x is positive and is less than 171 then Gamma(x) is returned and    //
    //     if x > 171 then DBL_MAX is returned.                                   //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x, g, nu;                                                  //
    //                                                                            //
    //     g = xLarge_x( x, nu);                                                  //
    ////////////////////////////////////////////////////////////////////////////////
    static long double xLarge_x(long double x, long double nu)
    {
       long double temp = 1.0L / nu;
       long double sum = temp;
       long double coef;
       int i = 0;
       int n;

       n = (int)(x - nu - 1.0L) + 1;
       for (i = 1; i < n; i++) {
          temp *= x / (nu + i);
          sum += temp;
       }
       if ( nu <= Gamma_Function_Max_Arg() ) {
          coef = powl(x, nu) * expl(-x) / xGamma_Function(nu);
          return xMedium_x(x, nu + n) + coef * sum;
       } else {
          return expl(logl(sum) + nu * logl(x) - x - xLn_Gamma_Function(nu)) +
                                                            xMedium_x(x, nu + n);
       }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // File: factorial.c                                                          //
    // Routine(s):                                                                //
    //    Factorial                                                               //
    //    xFactorial                                                              //
    //    Factorial_Max_Arg                                                       //
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    //  Description:                                                              //
    //     The functions Factorial(n) and xFactorial(n) return n!                 //
    //     for 0 <= n <= Factorial_Max_Arg().  If n > Factorial_Max_Arg(),        //
    //     then DBL_MAX is returned and if n < 0, then 0 is returned.             //
    //     The function Factorial_Max_Arg() returns the maximum argument to the   //
    //     functions Factorial() and xFactorial().                                //
    ////////////////////////////////////////////////////////////////////////////////

    //#include <float.h>                         // required for DBL_MAX

    //                         Internally Defined Routines                        //
    /*
    double Factorial(int n);
    long double xFactorial(int n);
    int Factorial_Max_Arg( void );
    */
    //                         Internally Defined Constants                       //

    static long double const factorials[] = {
           1.000000000000000000000e+0L,          //   0!
           1.000000000000000000000e+0L,          //   1!
           2.000000000000000000000e+0L,          //   2!
           6.000000000000000000000e+0L,          //   3!
           2.400000000000000000000e+1L,          //   4!
           1.200000000000000000000e+2L,          //   5!
           7.200000000000000000000e+2L,          //   6!
           5.040000000000000000000e+3L,          //   7!
           4.032000000000000000000e+4L,          //   8!
           3.628800000000000000000e+5L,          //   9!
           3.628800000000000000000e+6L,          //  10!
           3.991680000000000000000e+7L,          //  11!
           4.790016000000000000000e+8L,          //  12!
           6.227020800000000000000e+9L,          //  13!
           8.717829120000000000000e+10L,         //  14!
           1.307674368000000000000e+12L,         //  15!
           2.092278988800000000000e+13L,         //  16!
           3.556874280960000000000e+14L,         //  17!
           6.402373705728000000000e+15L,         //  18!
           1.216451004088320000000e+17L,         //  19!
           2.432902008176640000000e+18L,         //  20!
           5.109094217170944000000e+19L,         //  21!
           1.124000727777607680000e+21L,         //  22!
           2.585201673888497664000e+22L,         //  23!
           6.204484017332394393600e+23L,         //  24!
           1.551121004333098598400e+25L,         //  25!
           4.032914611266056355840e+26L,         //  26!
           1.088886945041835216077e+28L,         //  27!
           3.048883446117138605015e+29L,         //  28!
           8.841761993739701954544e+30L,         //  29!
           2.652528598121910586363e+32L,         //  30!
           8.222838654177922817726e+33L,         //  31!
           2.631308369336935301672e+35L,         //  32!
           8.683317618811886495518e+36L,         //  33!
           2.952327990396041408476e+38L,         //  34!
           1.033314796638614492967e+40L,         //  35!
           3.719933267899012174680e+41L,         //  36!
           1.376375309122634504632e+43L,         //  37!
           5.230226174666011117600e+44L,         //  38!
           2.039788208119744335864e+46L,         //  39!
           8.159152832478977343456e+47L,         //  40!
           3.345252661316380710817e+49L,         //  41!
           1.405006117752879898543e+51L,         //  42!
           6.041526306337383563736e+52L,         //  43!
           2.658271574788448768044e+54L,         //  44!
           1.196222208654801945620e+56L,         //  45!
           5.502622159812088949850e+57L,         //  46!
           2.586232415111681806430e+59L,         //  47!
           1.241391559253607267086e+61L,         //  48!
           6.082818640342675608723e+62L,         //  49!
           3.041409320171337804361e+64L,         //  50!
           1.551118753287382280224e+66L,         //  51!
           8.065817517094387857166e+67L,         //  52!
           4.274883284060025564298e+69L,         //  53!
           2.308436973392413804721e+71L,         //  54!
           1.269640335365827592597e+73L,         //  55!
           7.109985878048634518540e+74L,         //  56!
           4.052691950487721675568e+76L,         //  57!
           2.350561331282878571829e+78L,         //  58!
           1.386831185456898357379e+80L,         //  59!
           8.320987112741390144276e+81L,         //  60!
           5.075802138772247988009e+83L,         //  61!
           3.146997326038793752565e+85L,         //  62!
           1.982608315404440064116e+87L,         //  63!
           1.268869321858841641034e+89L,         //  64!
           8.247650592082470666723e+90L,         //  65!
           5.443449390774430640037e+92L,         //  66!
           3.647111091818868528825e+94L,         //  67!
           2.480035542436830599601e+96L,         //  68!
           1.711224524281413113725e+98L,         //  69!
           1.197857166996989179607e+100L,        //  70!
           8.504785885678623175212e+101L,        //  71!
           6.123445837688608686152e+103L,        //  72!
           4.470115461512684340891e+105L,        //  73!
           3.307885441519386412260e+107L,        //  74!
           2.480914081139539809195e+109L,        //  75!
           1.885494701666050254988e+111L,        //  76!
           1.451830920282858696341e+113L,        //  77!
           1.132428117820629783146e+115L,        //  78!
           8.946182130782975286851e+116L,        //  79!
           7.156945704626380229481e+118L,        //  80!
           5.797126020747367985880e+120L,        //  81!
           4.753643337012841748421e+122L,        //  82!
           3.945523969720658651190e+124L,        //  83!
           3.314240134565353266999e+126L,        //  84!
           2.817104114380550276949e+128L,        //  85!
           2.422709538367273238177e+130L,        //  86!
           2.107757298379527717214e+132L,        //  87!
           1.854826422573984391148e+134L,        //  88!
           1.650795516090846108122e+136L,        //  89!
           1.485715964481761497310e+138L,        //  90!
           1.352001527678402962552e+140L,        //  91!
           1.243841405464130725548e+142L,        //  92!
           1.156772507081641574759e+144L,        //  93!
           1.087366156656743080274e+146L,        //  94!
           1.032997848823905926260e+148L,        //  95!
           9.916779348709496892096e+149L,        //  96!
           9.619275968248211985333e+151L,        //  97!
           9.426890448883247745626e+153L,        //  98!
           9.332621544394415268170e+155L,        //  99!
           9.332621544394415268170e+157L,        // 100!
           9.425947759838359420852e+159L,        // 101!
           9.614466715035126609269e+161L,        // 102!
           9.902900716486180407547e+163L,        // 103!
           1.029901674514562762385e+166L,        // 104!
           1.081396758240290900504e+168L,        // 105!
           1.146280563734708354534e+170L,        // 106!
           1.226520203196137939352e+172L,        // 107!
           1.324641819451828974500e+174L,        // 108!
           1.443859583202493582205e+176L,        // 109!
           1.588245541522742940425e+178L,        // 110!
           1.762952551090244663872e+180L,        // 111!
           1.974506857221074023537e+182L,        // 112!
           2.231192748659813646597e+184L,        // 113!
           2.543559733472187557120e+186L,        // 114!
           2.925093693493015690688e+188L,        // 115!
           3.393108684451898201198e+190L,        // 116!
           3.969937160808720895402e+192L,        // 117!
           4.684525849754290656574e+194L,        // 118!
           5.574585761207605881323e+196L,        // 119!
           6.689502913449127057588e+198L,        // 120!
           8.094298525273443739682e+200L,        // 121!
           9.875044200833601362412e+202L,        // 122!
           1.214630436702532967577e+205L,        // 123!
           1.506141741511140879795e+207L,        // 124!
           1.882677176888926099744e+209L,        // 125!
           2.372173242880046885677e+211L,        // 126!
           3.012660018457659544810e+213L,        // 127!
           3.856204823625804217357e+215L,        // 128!
           4.974504222477287440390e+217L,        // 129!
           6.466855489220473672507e+219L,        // 130!
           8.471580690878820510985e+221L,        // 131!
           1.118248651196004307450e+224L,        // 132!
           1.487270706090685728908e+226L,        // 133!
           1.992942746161518876737e+228L,        // 134!
           2.690472707318050483595e+230L,        // 135!
           3.659042881952548657690e+232L,        // 136!
           5.012888748274991661035e+234L,        // 137!
           6.917786472619488492228e+236L,        // 138!
           9.615723196941089004197e+238L,        // 139!
           1.346201247571752460588e+241L,        // 140!
           1.898143759076170969429e+243L,        // 141!
           2.695364137888162776589e+245L,        // 142!
           3.854370717180072770522e+247L,        // 143!
           5.550293832739304789551e+249L,        // 144!
           8.047926057471991944849e+251L,        // 145!
           1.174997204390910823948e+254L,        // 146!
           1.727245890454638911203e+256L,        // 147!
           2.556323917872865588581e+258L,        // 148!
           3.808922637630569726986e+260L,        // 149!
           5.713383956445854590479e+262L,        // 150!
           8.627209774233240431623e+264L,        // 151!
           1.311335885683452545607e+267L,        // 152!
           2.006343905095682394778e+269L,        // 153!
           3.089769613847350887959e+271L,        // 154!
           4.789142901463393876336e+273L,        // 155!
           7.471062926282894447084e+275L,        // 156!
           1.172956879426414428192e+278L,        // 157!
           1.853271869493734796544e+280L,        // 158!
           2.946702272495038326504e+282L,        // 159!
           4.714723635992061322407e+284L,        // 160!
           7.590705053947218729075e+286L,        // 161!
           1.229694218739449434110e+289L,        // 162!
           2.004401576545302577600e+291L,        // 163!
           3.287218585534296227263e+293L,        // 164!
           5.423910666131588774984e+295L,        // 165!
           9.003691705778437366474e+297L,        // 166!
           1.503616514864999040201e+300L,        // 167!
           2.526075744973198387538e+302L,        // 168!
           4.269068009004705274939e+304L,        // 169!
           7.257415615307998967397e+306L         // 170!
                                            };

    static const int N = sizeof(factorials) / sizeof(long double);

    ////////////////////////////////////////////////////////////////////////////////
    // double Factorial( int n )                                                  //
    //                                                                            //
    //  Description:                                                              //
    //     This function computes n! for 0 <= n <= Factorial_Max_Arg().  If       //
    //     n > Factorial_Max_Arg(), then DBL_MAX is returned and if n < 0, then   //
    //     0 is returned.                                                         //
    //                                                                            //
    //  Arguments:                                                                //
    //     int    n   Argument of the Factorial function.                         //
    //                                                                            //
    //  Return Values:                                                            //
    //     If n is negative, then 0 is returned.  If n > Factorial_Max_Arg(),     //
    //     then DBL_MAX is returned.  If 0 <= n <= Factorial_Max_Arg(), then      //
    //     n! is returned.                                                        //
    //                                                                            //
    //  Example:                                                                  //
    //     int n;                                                                 //
    //     double x;                                                              //
    //                                                                            //
    //     x = Factorial( n );                                                    //
    ////////////////////////////////////////////////////////////////////////////////
    double Factorial(int n) {

                   // For a negative argument (n < 0) return 0.0 //

       if ( n < 0 ) return 0.0;


               // For a large postive argument (n >= N) return DBL_MAX //

       if ( n >= N ) return DBL_MAX;

                              // Otherwise return n!. //

       return (double) factorials[n];
    }


    ////////////////////////////////////////////////////////////////////////////////
    // long double xFactorial( int n )                                            //
    //                                                                            //
    //  Description:                                                              //
    //     This function computes n! for 0 <= n <= Factorial_Max_Arg().  If       //
    //     n > Factorial_Max_Arg(), then DBL_MAX is returned and if n < 0, then   //
    //     0 is returned.                                                         //
    //                                                                            //
    //  Arguments:                                                                //
    //     int    n   Argument of the Factorial function.                         //
    //                                                                            //
    //  Return Values:                                                            //
    //     If n is negative, then 0 is returned.  If n > Factorial_Max_Arg(),     //
    //     then DBL_MAX is returned.  If 0 <= n <= Factorial_Max_Arg(), then      //
    //     n! is returned.                                                        //
    //                                                                            //
    //  Example:                                                                  //
    //     int n;                                                                 //
    //     long double x;                                                         //
    //                                                                            //
    //     x = xFactorial( n );                                                   //
    ////////////////////////////////////////////////////////////////////////////////
    long double xFactorial(int n) {

                   // For a negative argument (n < 0) return 0.0 //

       if ( n < 0 ) return 0.0L;


               // For a large postive argument (n >= N) return DBL_MAX //

       if ( n >= N ) return  (long double) DBL_MAX;

                              // Otherwise return n!. //

       return factorials[n];
    }


    ////////////////////////////////////////////////////////////////////////////////
    // int Factorial_Max_Arg( void )                                              //
    //                                                                            //
    //  Description:                                                              //
    //     This function returns the maximum argument of the Factorial function   //
    //     for which a number < DBL_MAX is returned.                              //
    //                                                                            //
    //  Arguments:                                                                //
    //     none                                                                   //
    //                                                                            //
    //  Return Values:                                                            //
    //     N-1                                                                    //
    //                                                                            //
    //  Example:                                                                  //
    //     int x;                                                                 //
    //                                                                            //
    //     x = Factorial_Max_Arg();                                               //
    ////////////////////////////////////////////////////////////////////////////////

    int Factorial_Max_Arg( void ) { return N - 1; }


    ////////////////////////////////////////////////////////////////////////////////
    // File: gamma_function.c                                                     //
    // Routine(s):                                                                //
    //    Gamma_Function                                                          //
    //    xGamma_Function                                                         //
    //    Gamma_Function_Max_Arg                                                  //
    //    xGamma_Function_Max_Arg                                                 //
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    //  Description:                                                              //
    //     The Gamma function of x for real x > 0 is defined as:                  //
    //               Gamma(x) = Integral[0,inf] t^(x-1) exp(-t) dt                //
    //     and analytically continued in the complex plane with simple poles at   //
    //     the nonpositive integers, i.e. the Gamma function is a meromorphic     //
    //     function with simple poles at the nonpositive integers.                //
    //     For real x < 0, the Gamma function satisfies the reflection equation:  //
    //                Gamma(x) = pi / ( sin(pi*x) * Gamma(1-x) ).                 //
    //                                                                            //
    //     The functions Gamma_Function() and xGamma_Function() return the Gamma  //
    //     function evaluated at x for x real.                                    //
    //                                                                            //
    //     The function Gamma_Function_Max_Arg() returns the maximum argument of  //
    //     the Gamma function for arguments > 1 and return values of type double. //
    //                                                                            //
    //     The function xGamma_Function_Max_Arg() returns the maximum argument of //
    //     the Gamma function for arguments > 1 and return values of type long    //
    //     double.                                                                //
    ////////////////////////////////////////////////////////////////////////////////
    //                         Internally Defined Constants                       //

    static long double const e =  2.71828182845904523536028747L;
    static long double const g =  9.65657815377331589457187L;
    static long double const exp_g_o_sqrt_2pi = +6.23316569877722552586386e+3L;
    static double max_double_arg = 171.0;
    static long double max_long_double_arg = 1755.5L;
    static const long double log_sqrt_2pi = 9.18938533204672741780329736e-1L;

    static long double const a[] = {
                                     +1.14400529453851095667309e+4L,
                                     -3.23988020152318335053598e+4L,
                                     +3.50514523505571666566083e+4L,
                                     -1.81641309541260702610647e+4L,
                                     +4.63232990536666818409138e+3L,
                                     -5.36976777703356780555748e+2L,
                                     +2.28754473395181007645155e+1L,
                                     -2.17925748738865115560082e-1L,
                                     +1.08314836272589368860689e-4L
                                  };

    ////////////////////////////////////////////////////////////////////////////////
    // double Gamma_Function( double x )                                          //
    //                                                                            //
    //  Description:                                                              //
    //     This function uses Lanczos' expression to calculate Gamma(x) for real  //
    //     x, where -(max_double_arg - 1) < x < max_double_arg.                   //
    //     Note the Gamma function is meromorphic in the complex plane and has    //
    //     poles at the nonpositive integers.                                     //
    //     Tests for x a positive integer or a half positive integer give a       //
    //     maximum absolute relative error of about 1.9e-16.                      //
    //                                                                            //
    //     If x > max_double_arg, then one should either use xGamma_Function(x)   //
    //     or calculate lnGamma(x).                                               //
    //     Note that for x < 0, ln (Gamma(x)) may be a complex number.            //
    //                                                                            //
    //  Arguments:                                                                //
    //     double x   Argument of the Gamma function.                             //
    //                                                                            //
    //  Return Values:                                                            //
    //     If x is positive and is less than max_double_arg then Gamma(x) is      //
    //     returned and if x > max_double_arg then DBL_MAX is returned.  If x is  //
    //     a nonpositive integer i.e. x is a pole, then DBL_MAX is returned       //
    //     ( note that Gamma(x) changes sign on each side of the pole).  If x is  //
    //     nonpositive nonintegral, then if Gamma(x) > DBL_MAX, then DBL_MAX is   //
    //     returned and if Gamma(x) < -DBL_MAX, then -DBL_MAX is returned.        //
    //                                                                            //
    //  Example:                                                                  //
    //     double x, g;                                                           //
    //                                                                            //
    //     g = Gamma_Function( x );                                               //
    ////////////////////////////////////////////////////////////////////////////////
    double Gamma_Function(double x)
    {
       long double g;

       if ( x > max_double_arg ) return DBL_MAX;
       g = xGamma_Function( (long double) x);
       if (fabsl(g) < DBL_MAX) return (double) g;
       return (g < 0.0L) ? -DBL_MAX : DBL_MAX;

    }


    ////////////////////////////////////////////////////////////////////////////////
    // long double xGamma_Function( long double x )                               //
    //                                                                            //
    //  Description:                                                              //
    //     This function uses Lanczos' expression to calculate Gamma(x) for real  //
    //     x, where -(max_long_double_arg - 1) < x < max_long_double_arg.         //
    //     Note the Gamma function is meromorphic in the complex plane and has    //
    //     poles at the nonpositive integers.                                     //
    //     Tests for x a positive integer or a half positive integer give a       //
    //     maximum absolute relative error of about 3.5e-16.                      //
    //                                                                            //
    //     If x > max_long_double_arg, then one should use lnGamma(x).            //
    //     Note that for x < 0, ln (Gamma(x)) may be a complex number.            //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Argument of the Gamma function.                        //
    //                                                                            //
    //  Return Values:                                                            //
    //     If x is positive and is less than max_long_double_arg, then Gamma(x)   //
    //     is returned and if x > max_long_double_arg, then LDBL_MAX is returned. //
    //     If x is a nonpositive integer i.e. x is a pole, then LDBL_MAX is       //
    //     returned ( note that Gamma(x) changes sign on each side of the pole).  //
    //     If x is nonpositive nonintegral, then if x > -(max_long_double_arg + 1)//
    //     then Gamma(x) is returned otherwise 0.0 is returned.                   //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x, g;                                                      //
    //                                                                            //
    //     g = xGamma_Function( x );                                              //
    ////////////////////////////////////////////////////////////////////////////////
    long double xGamma_Function(long double x)
    {
       long double sin_x;
       long double rg;
       long int ix;

       // For a positive argument (x > 0)                 //
       //    if x <= max_long_double_arg return Gamma(x)  //
       //    otherwise      return LDBL_MAX.              //

       if  (x > 0.0L) {
          if (x <= max_long_double_arg) return xGamma(x);
          else return LDBL_MAX;
       }

       // For a nonpositive argument (x <= 0) //
       //    if x is a pole return LDBL_MAX   //

       if ( x > -(long double)LONG_MAX) {
          ix = (long int) x;
          if ( x == (long double) ix) return LDBL_MAX;
       }
       sin_x = sinl(PI * x);
       if ( sin_x == 0.0L ) return LDBL_MAX;

              // if x is not a pole and x < -(max_long_double_arg - 1) //
              //                                     then return 0.0L  //

       if ( x < -(max_long_double_arg - 1.0L) ) return 0.0L;

                // if x is not a pole and x >= -(max_long_double - 1) //
                //                               then return Gamma(x) //

       rg = xGamma(1.0L - x) * sin_x / PI;
       if ( rg != 0.0L ) return (1.0L / rg);
       return LDBL_MAX;
    }


    ////////////////////////////////////////////////////////////////////////////////
    // static long double xGamma( long double x )                                 //
    //                                                                            //
    //  Description:                                                              //
    //     This function uses Lanczos' expression to calculate Gamma(x) for real  //
    //     x, where 0 < x <= 900. For 900 < x < 1755.5, the duplication formula   //
    //     is used.                                                               //
    //     The major source of relative error is in the use of the c library      //
    //     function powl().  The results have a relative error of about 10^-16.   //
    //     except near x = 0.                                                     //
    //                                                                            //
    //     If x > 1755.5, then one should calculate lnGamma(x).                   //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Argument of the Gamma function.                        //
    //                                                                            //
    //  Return Values:                                                            //
    //     If x is positive and is less than 1755.5 then Gamma(x) is returned and //
    //     if x > 1755.5 then LDBL_MAX is returned.                               //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x;                                                         //
    //     long double g;                                                         //
    //                                                                            //
    //     g = xGamma_Function( x );                                              //
    ////////////////////////////////////////////////////////////////////////////////
    static long double xGamma(long double x)
    {

       long double xx = (x < 1.0L) ? x + 1.0L : x;
       long double temp;
       int const n = sizeof(a) / sizeof(long double);
       int i;

       if (x > 1755.5L) return LDBL_MAX;

       if (x > 900.0L) return Duplication_Formula(x);

       temp = 0.0L;
       for (i = n-1; i >= 0; i--) {
          temp += ( a[i] / (xx + (long double) i) );
       }
       temp += 1.0L;
       temp *= ( powl((g + xx - 0.5L) / e, xx - 0.5L) / exp_g_o_sqrt_2pi );
       return (x < 1.0L) ?  temp / x : temp;
    }


    ////////////////////////////////////////////////////////////////////////////////
    // static long double Duplication_Formula(long double two_x)                  //
    //                                                                            //
    //  Description:                                                              //
    //     This function returns the Gamma(two_x) using the duplication formula   //
    //     Gamma(2x) = (2^(2x-1) / sqrt(pi)) Gamma(x) Gamma(x+1/2).               //
    //                                                                            //
    //  Arguments:                                                                //
    //     none                                                                   //
    //                                                                            //
    //  Return Values:                                                            //
    //     Gamma(two_x)                                                           //
    //                                                                            //
    //  Example:                                                                  //
    //     long double two_x, g;                                                  //
    //                                                                            //
    //     g = Duplication_Formula(two_x);                                        //
    ////////////////////////////////////////////////////////////////////////////////
    static long double Duplication_Formula( long double two_x )
    {
       long double x = 0.5L * two_x;
       long double g;
       int n = (int) two_x - 1;

       g = powl(2.0L, two_x - 1.0L - (long double) n);
       g = ldexpl(g,n);
       g /= sqrt(PI);
       g *= xGamma_Function(x);
       g *= xGamma_Function(x + 0.5L);

       return g;
    }


    ////////////////////////////////////////////////////////////////////////////////
    // double Gamma_Function_Max_Arg( void )                                      //
    //                                                                            //
    //  Description:                                                              //
    //     This function returns the maximum argument of Gamma_Function for which //
    //     a number < DBL_MAX is returned, for arguments greater than 1.          //
    //                                                                            //
    //  Arguments:                                                                //
    //     none                                                                   //
    //                                                                            //
    //  Return Values:                                                            //
    //     max_double_arg (171.0).                                                //
    //                                                                            //
    //  Example:                                                                  //
    //     double x;                                                              //
    //                                                                            //
    //     x = Gamma_Function_Max_Arg();                                          //
    ////////////////////////////////////////////////////////////////////////////////
    double Gamma_Function_Max_Arg( void ) { return max_double_arg; }


    ////////////////////////////////////////////////////////////////////////////////
    // long double xGamma_Function_Max_Arg( void )                                //
    //                                                                            //
    //  Description:                                                              //
    //     This function returns the maximum argument of Gamma_Function for which //
    //     a number < LDBL_MAX is returned, for arguments greater than 1.         //
    //                                                                            //
    //  Arguments:                                                                //
    //     none                                                                   //
    //                                                                            //
    //  Return Values:                                                            //
    //     max_long_double_arg (1755.5).                                          //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x;                                                         //
    //                                                                            //
    //     x = xGamma_Function_Max_Arg();                                         //
    ////////////////////////////////////////////////////////////////////////////////
    long double xGamma_Function_Max_Arg( void ) { return max_long_double_arg; }

    ////////////////////////////////////////////////////////////////////////////////
    // double Incomplete_Gamma_Function(double x, double nu)                      //
    //                                                                            //
    //  Description:                                                              //
    //     The incomplete gamma function is defined as the integral from 0 to x   //
    //     of the integrand t^(nu-1) exp(-t) dt.  The parameter nu is sometimes   //
    //     referred to as the shape parameter.                                    //
    //                                                                            //
    //  Arguments:                                                                //
    //     double x   Upper limit of the integral with integrand given above.     //
    //     double nu  The shape parameter of the incomplete gamma function.       //
    //                                                                            //
    //  Return Values:                                                            //
    //     The incomplete gamma function Integral[0,x] t^(nu-1) exp(-t) dt.       //
    //                                                                            //
    //  Example:                                                                  //
    //     double x, g, nu;                                                       //
    //                                                                            //
    //     g = Incomplete_Gamma_Function( x, nu );                                //
    ////////////////////////////////////////////////////////////////////////////////
    double Incomplete_Gamma_Function(double x, double nu) {
       return (double) xIncomplete_Gamma_Function((long double)x, (long double)nu);
    }


    ////////////////////////////////////////////////////////////////////////////////
    // long double xIncomplete_Gamma_Function(long double x, long double nu)      //
    //                                                                            //
    //  Description:                                                              //
    //     The incomplete gamma function is defined as the integral from 0 to x   //
    //     of the integrand t^(nu-1) exp(-t) dt.  The parameter nu is sometimes   //
    //     referred to as the shape parameter.                                    //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Upper limit of the integral with integrand given above.//
    //     long double nu  The shape parameter of the incomplete gamma function.  //
    //                                                                            //
    //  Return Values:                                                            //
    //     The incomplete gamma function Integral[0,x] t^(nu-1) exp(-t) dt.       //
    //                                                                            //
    //  Example:                                                                  //
    //     long double x, g, nu;                                                  //
    //                                                                            //
    //     g = xIncomplete_Gamma_Function( x, nu );                               //
    ////////////////////////////////////////////////////////////////////////////////

    long double xIncomplete_Gamma_Function(long double x, long double nu) {

       if ( x == 0.0L ) return 0.0L;
       if ( nu <= Gamma_Function_Max_Arg() )
          return xEntire_Incomplete_Gamma_Function(x,nu) * xGamma_Function(nu);
       else
          return expl(logl(xEntire_Incomplete_Gamma_Function(x,nu))
                                                      + xLn_Gamma_Function(nu));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // double Ln_Gamma_Function( double x )                                       //
    //                                                                            //
    //  Description:                                                              //
    //     This function calculates the natural log of Gamma(x) for positive real //
    //     x.                                                                     //
    //     Assuming that Gamma_Function_Max_Arg() = 171, then                     //
    //     If 0 < x <= 171, then ln(gamma(x)) is calculated by taking the natural //
    //     log of the result from Gamma_Function(x).  If x > 171, then            //
    //     ln(gamma(x)) is calculated using the asymptotic expansion              //
    //         ln(gamma(x)) ~ ln(2sqrt(2pi)) - x + (x - 1/2) ln x +               //
    //                        Sum B[2j] / [ 2j * (2j-1) * x^(2j-1) ], summed over //
    //     j from 1 to 3 and where B[2j] is the 2j-th Bernoulli number.           //
    //                                                                            //
    //  Arguments:                                                                //
    //     double x   Argument of the ln Gamma function. The argument x must be   //
    //                positive.                                                   //
    //                                                                            //
    //  Return Values:                                                            //
    //     ln(Gamma(x)) where x > 0.                                              //
    //                                                                            //
    //  Example:                                                                  //
    //     double x, g;                                                           //
    //                                                                            //
    //     g = Ln_Gamma_Function( x );                                            //
    ////////////////////////////////////////////////////////////////////////////////

    double Ln_Gamma_Function(double x)
    {

           // For a positive argument, 0 < x <= Gamma_Function_Max_Arg() //
           // then  return log Gamma(x).                                 //

       if (x <= Gamma_Function_Max_Arg()) return log(Gamma_Function(x));

        // otherwise return result from asymptotic expansion of ln Gamma(x). //

       return (double) xLnGamma_Asymptotic_Expansion( (long double) x );
    }


    /*//////////////////////////////////////////////////////////////////////////////
    // long double xLn_Gamma_Function( long double x )                            //
    //                                                                            //
    //  Description:                                                              //
    //     This function calculates the natural log of Gamma(x) for positive real //
    //     x.                                                                     //
    //     Assuming that Gamma_Function_Max_Arg() = 171, then                     //
    //     If 0 < x <= 171, then ln(gamma(x)) is calculated by taking the natural //
    //     log of the result from Gamma_Function(x).  If x > 171, then            //
    //     ln(gamma(x)) is calculated using the asymptotic expansion              //
    //         ln(gamma(x)) ~ ln(2sqrt(2pi)) - x + (x - 1/2) ln x +               //
    //                        Sum B[2j] / [ 2j * (2j-1) * x^(2j-1) ], summed over //
    //     j from 1 to 3 and where B[2j] is the 2j-th Bernoulli number.           //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Argument of the ln Gamma function. The argument x must //
    //                     be positive.                                           //
    //                                                                            //
    //  Return Values:                                                            //
    //     ln(Gamma(x)) where x > 0.                                              //
    //                                                                            //
    //  Example:                                                                  //
    //     double x;                                                              //
    //     long double g;                                                         //
    //                                                                            //
    //     g = xLn_Gamma_Function( x );                                           //
    //////////////////////////////////////////////////////////////////////////////*/
    long double xLn_Gamma_Function(long double x)
    {

           // For a positive argument, 0 < x <= Gamma_Function_Max_Arg() //
           // then  return log Gamma(x).                                 //

       if (x <= Gamma_Function_Max_Arg()) return logl(xGamma_Function(x));

        // otherwise return result from asymptotic expansion of ln Gamma(x). //

       return xLnGamma_Asymptotic_Expansion( x );
    }


    /*//////////////////////////////////////////////////////////////////////////////
    // static long double xLnGamma_Asymptotic_Expansion( long double x )          //
    //                                                                            //
    //  Description:                                                              //
    //     This function estimates log(gamma(x)) by evaluating the asymptotic     //
    //     expression:                                                            //
    //         ln(Gamma(x)) ~ ln(2sqrt(2pi)) - x + (x - 1/2) ln x +               //
    //                        Sum B[2j] / [ 2j * (2j-1) * x^(2j-1) ], summed over //
    //     j from 1 to 3 and where B[2j] is the 2j-th Bernoulli number.           //
    //                                                                            //
    //  Arguments:                                                                //
    //     long double x   Argument of the ln Gamma function. The argument x must //
    //                     be  positive.                                          //
    //                                                                            //
    //  Return Values:                                                            //
    //     ln(Gamma(x)) where x > Gamma_Function_Max_Arg()                        //
    //                                                                            //
    //  Example:                                                                  //
    //     double x;                                                              //
    //     long double g;                                                         //
    //                                                                            //
    //     g = xlnGamma_Asymptotic_Expansion( x );                                //
    //////////////////////////////////////////////////////////////////////////////*/

    //static long double const pi = 3.14159265358979323846264338L;

    // Bernoulli numbers B(2),B(4),B(6),...,B(20).  Only B(2),...,B(6) currently used.                                                                     //
    static const long double B[] = {   1.0L / (6 * 2 * 1),
                                      -1.0L / (30 * 4 * 3),
                                       1.0L / (42 * 6 * 5),
                                      -1.0L / (30 * 8 * 7),
                                       5.0L / (66 * 10 * 9),
                                    -691.0L / (2730 * 12 * 11),
                                       7.0L / (6 * 14 * 13),
                                   -3617.0L / (510 * 16 * 15),
                                   43867.0L / (796 * 18 * 17),
                                 -174611.0L / (330 * 20 * 19)
                               };

    static const int n = sizeof(B) / sizeof(long double);


    static long double xLnGamma_Asymptotic_Expansion( long double x )
    {
       const int  m = 3;
       long double term[3];
       long double sum = 0.0L;
       long double xx = x * x;
       long double xj = x;
       long double lngamma = log_sqrt_2pi - xj + (xj - 0.5L) * logl(xj);
       int i;

       for (i = 0; i < m; i++) { term[i] = B[i] / xj; xj *= xx; }
       for (i = m - 1; i >= 0; i--) sum += term[i];
       return lngamma + sum;
    }

    namespace gammaDistributions
    {
        double gammaNaturalLogarithm(double value)
        //Returns the value ln[Γ(xx)] for xx > 0.
        {
        //Internal arithmetic will be done in double precision, a nicety that you can omit if five-figure
        //accuracy is good enough.
            double x,y,tmp,series;
            static double coefficients[6]={76.18009172947146,-86.50532032941677,
            24.01409824083091,-1.231739572450155,
            0.1208650973866179e-2,-0.5395239384953e-5};
            int j;
            y=x=value;
            tmp=x+5.5;
            tmp -= (x+0.5)*log(tmp);
            series=1.000000000190015;
            for (j=0;j<=5;j++) series += coefficients[j]/++y;
            return -tmp+log(2.5066282746310005*series/x);
        }


        void gammaIncompleteP(double *gammaDevelopmentSeries, double alpha, double x, double *gammaLn)
        //Returns the incomplete gamma function P(a, x) evaluated by its series representation as gamser.
        //Also returns ln Γ(a) as gln.
        {
            //double gammaNaturalLogarithm(double value);


            //void nrerror(char error_text[]);
            int n;
            double sum,del,ap;
            *gammaLn=gammaNaturalLogarithm(alpha);
            if (x <= 0.0)
            {
                //if (x < 0.0) printf("x less than 0 in routine gammaIncompleteP");
                *gammaDevelopmentSeries=0.0;
                return;
            }
            else
            {
                ap=alpha;
                del=sum=1.0/alpha;
                for (n=1;n<=ITERATIONSMAX;n++)
                {
                    ++ap;
                    del *= x/ap;
                    sum += del;
                    if (fabs(del) < fabs(sum)*EPSTHRESHOLD)
                    {
                        *gammaDevelopmentSeries=sum*exp(-x+alpha*log(x)-(*gammaLn));
                        return;
                    }
                }
                //printf("a too large, ITERATIONSMAX too small in routine gammaIncompleteP");
                return;
            }
        }

        void gammaIncompleteComplementaryFunction(double *gammaComplementaryFunction, double alpha, double x, double *gammaLn)
        //Returns the incomplete gamma function Q(a, x) evaluated by its continued fraction representation
        //as gammcf. Also returns lnΓ(a) as gln.
        {
            //double gammaNaturalLogarithm(double value);
            //void nrerror(char error_text[]);
            int i;
            double an,b,c,d,del,h;
            *gammaLn=gammaNaturalLogarithm(alpha);
            b=x+1.0-alpha; //Set up for evaluating continued fraction by modified Lentz’s method (§5.2)with b0 = 0.
            c=1.0/FPMINIMUM;
            d=1.0/b;
            h=d;
            for (i=1;i<=ITERATIONSMAX;i++)
            { //Iterate to convergence.
                an = -i*(i-alpha);
                b += 2.0;
                d=an*d+b;
                if (fabs(d) < FPMINIMUM) d=FPMINIMUM;
                c=b+an/c;
                if (fabs(c) < FPMINIMUM) c=FPMINIMUM;
                d=1.0/d;
                del=d*c;
                h *= del;
                if (fabs(del-1.0) < EPSTHRESHOLD) break;
            }            
            *gammaComplementaryFunction=exp(-x+alpha*log(x)-(*gammaLn))*h; //Put factors in front.
        }

        double incompleteGamma(double alpha, double x, double *lnGammaValue)
        {
            /* this function returns
             * 1) the value of the normalized incomplete gamma function
             * 2) the natural logarithm of the complete gamma function
             * pay attention to the inputs: input variable x is actually beta*x or x/theta
             * written by Antonio Volta avolta@arpae.it
            */
            double gammaIncompleteCF;
            double gammaIncomplete;
            if (x > alpha + 1)
            {
                gammaIncompleteComplementaryFunction(&gammaIncompleteCF,alpha,x,lnGammaValue);
                gammaIncomplete = 1 - gammaIncompleteCF;
            }
            else
                gammaIncompleteP(&gammaIncomplete,alpha,x,lnGammaValue);

            return gammaIncomplete;
        }

        double incompleteGamma(double alpha, double x)
        {
            /* this function returns
             * 1) the value of the normalized incomplete gamma function
             * pay attention to the inputs: input variable x is actually beta*x or x/theta
             * written by Antonio Volta avolta@arpae.it
            */
            double gammaIncompleteCF;
            double gammaIncomplete;
            double lnGammaValue;
            if (x > alpha + 1)
            {
                gammaIncompleteComplementaryFunction(&gammaIncompleteCF,alpha,x,&lnGammaValue);
                gammaIncomplete = 1 - gammaIncompleteCF;
            }
            else
                gammaIncompleteP(&gammaIncomplete,alpha,x,&lnGammaValue);

            return gammaIncomplete;
        }

        bool gammaFitting(std::vector<float> &series, int n, double *beta, double *gamma,  double *pZero)
        {
            if (n<=0)
            {
                return false;
            }
            double sum = 0;
            double sumLog = 0;
            double alpha;
            int nAct = 0;
            double average = 0;

            // compute sums
            for (int i = 0; i<n; i++)
            {
                if (series[i] != NODATA)
                {
                    if (series[i] > 0)
                    {
                        sum = sum + series[i];
                        sumLog = sumLog + log(series[i]);
                        nAct = nAct + 1;
                    }
                    else
                    {
                        *pZero = *pZero + 1;
                    }
                }
            }
            if (nAct > 0)
            {
                average = sum / nAct;
            }

            if (nAct == 1)
            {
                // Bogus data array but do something reasonable
                *pZero = 0;
                alpha = 0;
                *gamma = 1;
                *beta = average;
            }
            else if (*pZero == n)
            {
                // They were all zeroes
                *pZero = 1;
                alpha = 0;
                *gamma = 1;
                *beta = average;
            }
            else
            {
                // Use MLE
                *pZero = *pZero/n;
                alpha = log(average) - sumLog / nAct;
                *gamma = (1 + sqrt(1 + 4 * alpha / 3)) / (4 * alpha);
                *beta = average / (*gamma);
            }

            return true;
        }

    }

    float standardGaussianInvCDF(float prob)
    {
        double  resul;
        resul = SQRT_2 * statistics::inverseTabulatedERF(2*prob -1);
        return resul;
    }

    /*
    Compute probability of a<=x using incomplete gamma parameters.
    Input:     beta, gamma (gamma parameters)
               pzero (probability of zero)
               x (value)
    Output:    GammaCDF (probability  a<=x)
    */
    float gammaCDF(float x, double beta, double gamma,  double pZero)
    {

        float gammaCDF = NODATA;

        if (x == NODATA || beta == NODATA || gamma == NODATA || pZero == NODATA || beta == 0)
        {
            return gammaCDF;
        }

        if (x <= 0)
        {
            gammaCDF = pZero;
        }
        else
        {
            gammaCDF = pZero + (1 - pZero) * gammaDistributions::incompleteGamma(gamma, x / beta);
        }
        return gammaCDF;

    }

    void probabilityWeightedMoments(std::vector<float> series, int n, std::vector<float> &probWeightedMoments, float a, float b, bool isBeta)
    {

        float f;
        std::vector<float> sum{0,0,0};

        if (a == 0 && b == 0)
        {
            // use unbiased estimator
            for (int i = 0; i < n; i++)
            {
                sum[0] = sum[0] + series[i];
                if (!isBeta)
                {
                    // compute alpha PWMs
                    sum[1] = sum[1] + series[i] * (n - i);
                    sum[2] = sum[2] + series[i] * (n - i) * (n - i - 1);
                }
                else
                {
                    // compute beta PWMs
                    sum[1] = sum[1] + series[i] * (i - 1);
                    sum[2] = sum[2] + series[i] * (i - 1) * (i - 2);
                }
            }
        }
        else
        {
            // use plotting-position (biased) estimator
            for (int i = 0; i < n; i++)
            {
                sum[0] = sum[0] + series[i];
                f = (i + a) / (n + b);
                if (!isBeta)
                {
                    // compute alpha PWMs
                    sum[1] = sum[1] + series[i] * (1 - f);
                    sum[2] = sum[2] + series[i] * (1 - f) * (1 - f);
                }
                else
                {
                    // compute beta PWMs
                    sum[1] = sum[1] + series[i] * f;
                    sum[2] = sum[2] + series[i] * f * f;
                 }
            }
        }

        probWeightedMoments[0] = sum[0] / n ;
        probWeightedMoments[1] = sum[1] / n / (n - 1) ;
        probWeightedMoments[2] = sum[2] / n / ((n - 1) * (n - 2)) ;

    }

    // Estimates the parameters of a log-logistic distribution function
    void logLogisticFitting(std::vector<float> probWeightedMoments, double *alpha, double *beta, double *gamma)
    {
        float g1;
        float g2;
        *gamma = (2 * probWeightedMoments[1] - probWeightedMoments[0]) / (6 * probWeightedMoments[1] - probWeightedMoments[0] - 6 * probWeightedMoments[2]);
        g1 = exp(Ln_Gamma_Function(1 + 1 / (*gamma)));
        g2 = exp(Ln_Gamma_Function(1 - 1 / (*gamma)));
        *alpha = (probWeightedMoments[0] - 2 * probWeightedMoments[1]) * (*gamma) / (g1 * g2);
        *beta = probWeightedMoments[0] - (*alpha) * g1 * g2;
    }

    // Gives the cumulative distribution function of input "myValue",
    // following a LogLogistic distribution
    float logLogisticCDF(float myValue, double alpha, double beta, double gamma)
    {
        float logLogisticCDF = 1 / (1 + (pow((alpha / (myValue - beta)), gamma)));

        return logLogisticCDF;
    }
