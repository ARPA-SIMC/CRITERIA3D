#ifndef OLIVO_H
#define OLIVO_H

    #include "coltura.h"

    class Olivo : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_inizioAttivita;
        Crit3DDate m_induzioneFiorale;
        Crit3DDate m_fioritura;
        Crit3DDate m_maturazione;
        double m_limiteGradiGiorno[2];

        std::vector<double> m_faseFenologica;

    public:

        Olivo()
        :	m_coltura(nullptr),
            m_inizioAttivita(Crit3DDate()),
            m_induzioneFiorale(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_faseFenologica(std::vector<double>())
        {
            m_limiteGradiGiorno[0] = 0.;
            m_limiteGradiGiorno[1] = 0.;
        }


        Olivo(const char* coltura, const double& limiteGradiGiorno0, const double& limiteGradiGiorno1)
        :	m_coltura(coltura),
            m_inizioAttivita(Crit3DDate()),
            m_induzioneFiorale(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_faseFenologica(std::vector<double>())
        {
            m_limiteGradiGiorno[0] = limiteGradiGiorno0;
            m_limiteGradiGiorno[1] = limiteGradiGiorno1;
        }

        ~Olivo()
        {
            m_faseFenologica.clear();
        }

        void InduzioneFiorale(const Stazione& stazione);
        void Fioritura(const Stazione& stazione);
        void Maturazione(const Stazione& stazione);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long& giorno, const Parametri& parametri);
    };

#endif


