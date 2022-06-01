#ifndef VITE_H
#define VITE_H

    #include "coltura.h"


    class Vite : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_inizioAttivita;
        Crit3DDate m_germogliamento;
        Crit3DDate m_fioritura;
        Crit3DDate m_invaiatura;
        Crit3DDate m_maturazione;
        double m_limiteGradiGiorno[4];

        std::vector<double> m_faseFenologica;

    public:

        Vite()
        :	m_coltura(nullptr),
            m_inizioAttivita(Crit3DDate()),
            m_germogliamento(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_invaiatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_faseFenologica(std::vector<double>())
        {
            m_limiteGradiGiorno[0] = 0.;
            m_limiteGradiGiorno[1] = 0.;
            m_limiteGradiGiorno[2] = 0.;
            m_limiteGradiGiorno[3] = 0.;
        }


        Vite(const char* coltura, double limiteGradiGiorno0, double limiteGradiGiorno1, double limiteGradiGiorno2, double limiteGradiGiorno3)
        :	m_coltura(coltura),
            m_inizioAttivita(Crit3DDate()),
            m_germogliamento(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_invaiatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_faseFenologica(std::vector<double>())
        {
            m_limiteGradiGiorno[0] = limiteGradiGiorno0;
            m_limiteGradiGiorno[1] = limiteGradiGiorno1;
            m_limiteGradiGiorno[2] = limiteGradiGiorno2;
            m_limiteGradiGiorno[3] = limiteGradiGiorno3;
        }

        ~Vite()
        {
            m_faseFenologica.clear();
        }

        // other functions
        double GradiGiorno(const long& i, const Stazione& stazione);

        void Germogliamento(const Stazione& stazione);
        void Fioritura(const Stazione& stazione);
        void Invaiatura(const Stazione& stazione);
        void Maturazione(const Stazione& stazione);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long& giorno, const Parametri& parametri);
    };

#endif




