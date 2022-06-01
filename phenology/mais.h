#ifndef MAIS_H
#define MAIS_H

    #include "coltura.h"

    class Mais : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_semina;
        Crit3DDate m_emergenza;
        Crit3DDate m_viraggioApicale;
        Crit3DDate m_fioritura;
        Crit3DDate m_maturazione;

        long m_classeFAO;

        double m_numeroFoglieEmerse;
        double m_numeroFoglieTotali;

        std::vector<double> m_limiteGradiGiorno;
        std::vector<double> m_faseFenologica;

    public:

        // ctors
        Mais()
        :	m_coltura(nullptr),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_viraggioApicale(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_classeFAO(100),
            m_numeroFoglieEmerse(0),
            m_numeroFoglieTotali(0),
            m_limiteGradiGiorno(std::vector<double>()),
            m_faseFenologica(std::vector<double>())
        {}

        Mais(const char* coltura, const long& varieta)
        :	m_coltura(coltura),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_viraggioApicale(Crit3DDate()),
            m_fioritura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_classeFAO(200 + varieta * 100),
            m_numeroFoglieEmerse(1.),
            m_numeroFoglieTotali(0),
            m_limiteGradiGiorno(std::vector<double>()),
            m_faseFenologica(std::vector<double>())
        {
            m_limiteGradiGiorno.push_back(568.);
            m_limiteGradiGiorno.push_back(671.);
            m_limiteGradiGiorno.push_back(767.);
            m_limiteGradiGiorno.push_back(795.);
            m_limiteGradiGiorno.push_back(813.);
            m_limiteGradiGiorno.push_back(850.);
        }

        // dtor
        ~Mais()
        {
            m_limiteGradiGiorno.clear();
            m_faseFenologica.clear();
        }


        // calcolo semplificato della temperatura del suolo
        double TemperaturaSuoloMinima(const long& giorno, const Stazione& stazione, const Parametri& parametri);
        double TemperaturaSuoloMassima(const long& giorno, const Stazione& stazione, const Parametri& parametri);

        long NumeroFoglieMinimo() { return 11 + m_classeFAO / 100; }

        void CalcoloNumeroFoglie(const long& giorno, Stazione& stazione, const Parametri& parametri);

        double TassoCrescitaPrimordi(const double& T) { return (-6.5E-4 - 1.39E-2*T + 3.72E-3*T*T - 7.2E-5*T*T*T); }
        double TassoCrescitaFoglie(const double& T) { return (9.97E-2 - 3.6E-2*T + 3.62E-3*T*T - 6.39E-5*T*T*T); }
        double EffettoTemperatura(const double& T) { return (13.6 - 1.89*T + 0.081*T*T - 0.001*T*T*T); }
        double CornHeatUnits(const double& Tx, const double& Tn) { return .5*(1.85*(Tx-10.) - 0.047*(Tx-10.)*(Tx-10.) + (Tn-4.4)); }

        // other function
        void Emergenza(const Stazione& stazione, const Parametri& parametri);
        void Viraggio(Stazione& stazione, const Parametri& parametri);
        void Fioritura(const Stazione& stazione, const Parametri& parametri);
        void Maturazione(const Stazione& stazione);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long &giorno, const Parametri& parametri);
    };

#endif
