#ifndef POMODORO_H
#define POMODORO_H

    #include "coltura.h"

    // Pomodoro da seme
    class Pomodoro : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_semina;
        Crit3DDate m_emergenza;
        Crit3DDate m_primoFiore;
        Crit3DDate m_invaiatura;
        Crit3DDate m_maturazione;
        long m_giorniSiccitosi;
        double m_minPHD;
        double m_maxPHD;
        double m_accumuloPrec;

        std::vector<double> m_faseFenologica;

    public:

        Pomodoro()
        :	m_coltura(nullptr),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_primoFiore(Crit3DDate()),
            m_invaiatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_giorniSiccitosi(0),
            m_minPHD(0.),
            m_maxPHD(0.),
            m_accumuloPrec(0.),
            m_faseFenologica(std::vector<double>())
        {}

        Pomodoro(const char* coltura, double minPHD, double maxPHD)
        :	m_coltura(coltura),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_primoFiore(Crit3DDate()),
            m_invaiatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_giorniSiccitosi(0),
            m_minPHD(minPHD),
            m_maxPHD(maxPHD),
            m_accumuloPrec(0.),
            m_faseFenologica(std::vector<double>())
        {}

        ~Pomodoro()
        {
            m_faseFenologica.clear();
        }


        double	Tday(const double& Tn, const double& Tx) { return Tn + .77 * ( Tx - Tn); }
        double	Tnyt(const double& Tn, const double& Tx) { return Tn + .19 * ( Tx - Tn); }

        // other functions
        double	Dday(Stazione& stazione, const long& i);
        double	Dday(Stazione& stazione, const long& i, const double& Ts);
        double	Dnyt(Stazione& stazione, const long& i, const double& Ts);

        void Emergenza(Stazione& stazione);
        void PrimoFiore(Stazione& stazione);
        void Invaiatura(Stazione& stazione);
        void Maturazione(Stazione& stazione);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long& giorno, const Parametri& parametri);
    };


    // Podomoro da trapianto
    class PomodoroTrapianto : public Pomodoro
    {
        const char* m_coltura;

        Crit3DDate m_trapianto;
        Crit3DDate m_primoFiore;
        Crit3DDate m_invaiatura;
        Crit3DDate m_maturazione;
        long m_giorniSiccitosi;
        double m_minPHD;
        double m_maxPHD;
        double m_accumuloPrec;

        std::vector<double> m_faseFenologica;

    public:

        PomodoroTrapianto()
        :	m_coltura(nullptr),
            m_trapianto(Crit3DDate()),
            m_primoFiore(Crit3DDate()),
            m_invaiatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_giorniSiccitosi(0),
            m_minPHD(0.),
            m_maxPHD(0.),
            m_accumuloPrec(0.),
            m_faseFenologica(std::vector<double>())
        {}

        PomodoroTrapianto(const char* coltura, double minPHD, double maxPHD)
        :	m_coltura(coltura),
            m_trapianto(Crit3DDate()),
            m_primoFiore(Crit3DDate()),
            m_invaiatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_giorniSiccitosi(0),
            m_minPHD(minPHD),
            m_maxPHD(maxPHD),
            m_accumuloPrec(0.),
            m_faseFenologica(std::vector<double>())
        {}

        ~PomodoroTrapianto()
        {
            m_faseFenologica.clear();
        }

        // other functions
        void PrimoFiore(Stazione& stazione);
        void Invaiatura(Stazione& stazione);
        void Maturazione(Stazione& stazione);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long& giorno, const Parametri& parametri);
    };

#endif
