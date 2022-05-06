#ifndef GIRASOLE_H
#define GIRASOLE_H

    #include "coltura.h"

    class Girasole : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_semina;
        Crit3DDate m_emergenza;
        Crit3DDate m_gemmaFiorale;
        Crit3DDate m_fioritura;
        Crit3DDate m_maturazione;
        Crit3DDate m_raccolta;

        double m_limiteGradiGiorno;
        double C1[2][4];
        double C2[2][4];
        double C3[2][4];
        double C4[2][4];

        std::vector<double> m_faseFenologica;

    public:

        Girasole()
        :	m_coltura(nullptr),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_gemmaFiorale(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_raccolta(Crit3DDate()),
            m_limiteGradiGiorno(0.),
            m_faseFenologica(std::vector<double>())
        {
            C1[0][0] = 0.;	C1[0][1] = 0.;	C1[0][2] = 0.;	C1[0][3] = 0.;
            C2[0][0] = 0.;	C2[0][1] = 0.;	C2[0][2] = 0.;	C2[0][3] = 0.;
            C3[0][0] = 0.;	C3[0][1] = 0.;	C3[0][2] = 0.;	C3[0][3] = 0.;
            C4[0][0] = 0.;	C4[0][1] = 0.;	C4[0][2] = 0.;	C4[0][3] = 0.;
            C1[1][0] = 0.;	C1[1][1] = 0.;	C1[1][2] = 0.;	C1[1][3] = 0.;
            C2[1][0] = 0.;	C2[1][1] = 0.;	C2[1][2] = 0.;	C2[1][3] = 0.;
            C3[1][0] = 0.;	C3[1][1] = 0.;	C3[1][2] = 0.;	C3[1][3] = 0.;
            C4[1][0] = 0.;	C4[1][1] = 0.;	C4[1][2] = 0.;	C4[1][3] = 0.;
        }

        Girasole(const char* coltura, const long& limiteGradiGiorno)
        :	m_coltura(coltura),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_gemmaFiorale(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_raccolta(Crit3DDate()),
            m_limiteGradiGiorno(limiteGradiGiorno),
            m_faseFenologica(std::vector<double>())
        {
            C1[0][0] = 0.00371;	C1[0][1] = 0.00331;	C1[0][2] = 0.00315;	C1[0][3] = 0.00287;
            C2[0][0] = 0.000092;C2[0][1] = 0.000071;C2[0][2] = 0.000067;C2[0][3] = 0.000058;
            C3[0][0] = 1.;		C3[0][1] = 0.831;	C3[0][2] = 0.789;	C3[0][3] = 0.702;
            C4[0][0] = 0.;		C4[0][1] = 0.00654;	C4[0][2] = 0.00658;	C4[0][3] = 0.00779;
            C1[1][0] = 0.00336;	C1[1][1] = 0.00348;	C1[1][2] = 0.00344;	C1[1][3] = 0.00306;
            C2[1][0] = 0.000057;C2[1][1] = 0.000067;C2[1][2] = 0.000069;C2[1][3] = 0.000054;
            C3[1][0] = 1.;		C3[1][1] = 1.062;	C3[1][2] = 1.059;	C3[1][3] = 0.916;
            C4[1][0] = 0.;		C4[1][1] = 0.00391;	C4[1][2] = 0.00549;	C4[1][3] = 0.00097;
        }

        ~Girasole()
        {
            m_faseFenologica.clear();
        }


        // other functions
        void Emergenza(const Stazione& stazione);
        void GemmaFiorale(const Stazione& stazione, const Parametri& parametri);
        void Fioritura(const Stazione& stazione, const Parametri& parametri);
        void Maturazione(const Stazione& stazione);
        void Raccolta(const Stazione& stazione);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long& giorno, const Parametri& parametri);
    };

#endif


