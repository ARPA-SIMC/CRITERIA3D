#ifndef SOIA_H
#define SOIA_H

    #include "coltura.h"

    class Soia : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_semina;
        Crit3DDate m_emergenza;
        Crit3DDate m_inizioFioritura;
        Crit3DDate m_fineFioritura;
        Crit3DDate m_ingrossamentoBaccello;
        Crit3DDate m_maturazione;
        Crit3DDate m_raccolta;

        double a0[4][3];
        double a1[4][3];
        double a2[4][3];
        double a3[4][3];
        double b0[4][3];
        double b1[4][3];
        double b2[4][3];

        std::vector<double> m_faseFenologica;

    public:

        Soia()
        :	m_coltura(nullptr),
            m_semina(Crit3DDate()),
            m_inizioFioritura(Crit3DDate()),
            m_fineFioritura(Crit3DDate()),
            m_ingrossamentoBaccello(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_raccolta(Crit3DDate()),
            m_faseFenologica(std::vector<double>())
        {
            a0[0][0] = 0.;	  a1[0][0] = 0.;		a2[0][0] = 0.;		  a3[0][0] = 1.;
            a0[1][0] = 8.02;  a1[1][0] = 2.37E-2;	a2[1][0] = -2.309E-3; a3[1][0] = 0.;
            a0[2][0] = 17.97; a1[2][0] = -3.493E-2; a2[2][0] = 1.503E-3;  a3[2][0] = 0.;
            a0[3][0] = 18.72; a1[3][0] = -2.095E-2; a2[3][0] = 0.;		  a3[3][0] = 0.;

            b0[0][0] = 10.9;  b1[0][0] = 2.15E-2;	b2[0][0] = -8.56E-4;
            b0[1][0] = 5.17;  b1[1][0] = 3.85E-2;	b2[1][0] = 0.;
            b0[2][0] = 14.64; b1[2][0] = 6.248E-2;	b2[2][0] = -4.15E-3;
            b0[3][0] = 11.89; b1[3][0] = 3.874E-2;	b2[3][0] = -1.852E-3;

            a0[0][1] = 0.;	  a1[0][1] = 0.;		a2[0][1] = 0.;		  a3[0][1] = 1.;
            a0[1][1] = 8.02;  a1[1][1] = 2.37E-2;   a2[1][1] = -2.309E-3; a3[1][1] = 0.;
            a0[2][1] = 17.97; a1[2][1] = -3.493E-2; a2[2][1] = 1.503E-3;  a3[2][1] = 0.;
            a0[3][1] = 18.72; a1[3][1] = -2.095E-2; a2[3][1] = 0.;		  a3[3][1] = 0.;

            b0[0][1] = 10.9;  b1[0][1] = 2.15E-2;   b2[0][1] = -8.56E-4;
            b0[1][1] = 5.17;  b1[1][1] = 3.85E-2;   b2[1][1] = 0.;
            b0[2][1] = 14.64; b1[2][1] = 6.248E-2;  b2[2][1] = -4.15E-3;
            b0[3][1] = 11.89; b1[3][1] = 3.874E-2;  b2[3][1] = -1.852E-3;

            a0[0][2] = 0.;	  a1[0][2] = 0.;		a2[0][2] = 0.; a3[0][2] = 1.;
            a0[1][2] = 8.72;  a1[1][2] = 2.435E-2;  a2[1][2] = 0.; a3[1][2] = 0.;
            a0[2][2] = 16.86; a1[2][2] = -4.256E-2; a2[2][2] = 0.; a3[2][2] = 0.;
            a0[3][2] = 17.91; a1[3][2] = -1.908E-2; a2[3][2] = 0.; a3[3][2] = 0.;

            b0[0][2] = 10.9;  b1[0][2] = 2.15E-2;	b2[0][2] = -8.56E-4;
            b0[1][2] = 3.5;	  b1[1][2] = 3.877E-2;	b2[1][2] = 0.;
            b0[2][2] = 13.89; b1[2][2] = 7.94E-2;	b2[2][2] = -5.032E-3;
            b0[3][2] = 11.03; b1[3][2] = 3.508E-2;	b2[3][2] = -1.307E-3;
        }

        Soia(const char* coltura)
        :	m_coltura(coltura),
            m_semina(Crit3DDate()),
            m_inizioFioritura(Crit3DDate()),
            m_fineFioritura(Crit3DDate()),
            m_ingrossamentoBaccello(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_raccolta(Crit3DDate()),
            m_faseFenologica(std::vector<double>())
        {
            a0[0][0] = 0.;	  a1[0][0] = 0.;		a2[0][0] = 0.;		  a3[0][0] = 1.;
            a0[1][0] = 8.02;  a1[1][0] = 2.37E-2;	a2[1][0] = -2.309E-3; a3[1][0] = 0.;
            a0[2][0] = 17.97; a1[2][0] = -3.493E-2; a2[2][0] = 1.503E-3;  a3[2][0] = 0.;
            a0[3][0] = 18.72; a1[3][0] = -2.095E-2; a2[3][0] = 0.;		  a3[3][0] = 0.;

            b0[0][0] = 10.9;  b1[0][0] = 2.15E-2;	b2[0][0] = -8.56E-4;
            b0[1][0] = 5.17;  b1[1][0] = 3.85E-2;	b2[1][0] = 0.;
            b0[2][0] = 14.64; b1[2][0] = 6.248E-2;	b2[2][0] = -4.15E-3;
            b0[3][0] = 11.89; b1[3][0] = 3.874E-2;	b2[3][0] = -1.852E-3;

            a0[0][1] = 0.;	  a1[0][1] = 0.;		a2[0][1] = 0.;		  a3[0][1] = 1.;
            a0[1][1] = 8.02;  a1[1][1] = 2.37E-2;   a2[1][1] = -2.309E-3; a3[1][1] = 0.;
            a0[2][1] = 17.97; a1[2][1] = -3.493E-2; a2[2][1] = 1.503E-3;  a3[2][1] = 0.;
            a0[3][1] = 18.72; a1[3][1] = -2.095E-2; a2[3][1] = 0.;		  a3[3][1] = 0.;

            b0[0][1] = 10.9;  b1[0][1] = 2.15E-2;   b2[0][1] = -8.56E-4;
            b0[1][1] = 5.17;  b1[1][1] = 3.85E-2;   b2[1][1] = 0.;
            b0[2][1] = 14.64; b1[2][1] = 6.248E-2;  b2[2][1] = -4.15E-3;
            b0[3][1] = 11.89; b1[3][1] = 3.874E-2;  b2[3][1] = -1.852E-3;

            a0[0][2] = 0.;	  a1[0][2] = 0.;		a2[0][2] = 0.; a3[0][2] = 1.;
            a0[1][2] = 8.72;  a1[1][2] = 2.435E-2;  a2[1][2] = 0.; a3[1][2] = 0.;
            a0[2][2] = 16.86; a1[2][2] = -4.256E-2; a2[2][2] = 0.; a3[2][2] = 0.;
            a0[3][2] = 17.91; a1[3][2] = -1.908E-2; a2[3][2] = 0.; a3[3][2] = 0.;

            b0[0][2] = 10.9;  b1[0][2] = 2.15E-2;	b2[0][2] = -8.56E-4;
            b0[1][2] = 3.5;	  b1[1][2] = 3.877E-2;	b2[1][2] = 0.;
            b0[2][2] = 13.89; b1[2][2] = 7.94E-2;	b2[2][2] = -5.032E-3;
            b0[3][2] = 11.03; b1[3][2] = 3.508E-2;	b2[3][2] = -1.307E-3;
        }

        ~Soia()
        {
            m_faseFenologica.clear();
        }

        double M(const long& i, const long& j, double& T, Stazione& stazione, const Parametri& parametri);

        // other functions
        void Emergenza(Stazione& stazione, const Parametri& parametri);
        void Fioritura(Stazione& stazione, const Parametri& parametri);
        void IngrossamentoBaccello(Stazione& stazione, const Parametri& parametri);
        void Maturazione(Stazione& stazione, const Parametri& parametri);
        void Raccolta(Stazione& stazione, const Parametri& parametri);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long& giorno, const Parametri& parametri);
    };

#endif
