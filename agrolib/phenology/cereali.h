#ifndef CEREALI_H
#define CEREALI_H

    #include "coltura.h"
    #include "math.h"
    #include <algorithm>

    // Classe per gli oggetti Cereali, Grano duro e Orzo
    class Cereali : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_seminaFittizia;
        Crit3DDate m_emergenzaFittizia;
        Crit3DDate m_semina;
        Crit3DDate m_emergenza;
        Crit3DDate m_viraggioApicale;
        Crit3DDate m_spigatura;
        Crit3DDate m_maturazione;

        double m_beta;
        double m_sigma;
        double m_numeroPrimordi;
        double m_numeroFoglieTotali;
        double m_numeroFoglieEmerse;
        double m_limiteGradiGiorno;
        double m_sogliaGradiGiorno;

        std::vector<double> m_faseFenologica;

    public:
        // ctors
        Cereali()
        :	m_coltura(nullptr),
            m_seminaFittizia(Crit3DDate()),
            m_emergenzaFittizia(Crit3DDate()),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_viraggioApicale(Crit3DDate()),
            m_spigatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_beta(0.),
            m_sigma(0.),
            m_numeroPrimordi(0.),
            m_numeroFoglieEmerse(0.),
            m_limiteGradiGiorno(0.),
            m_sogliaGradiGiorno(0.),
            m_faseFenologica(std::vector<double>())
        {}

        Cereali(const char* coltura, const double& beta, const double& sigma, const double& limiteGradiGiorno)
        :	m_coltura(coltura),
            m_seminaFittizia(Crit3DDate()),
            m_emergenzaFittizia(Crit3DDate()),
            m_semina(Crit3DDate()),
            m_emergenza(Crit3DDate()),
            m_viraggioApicale(Crit3DDate()),
            m_spigatura(Crit3DDate()),
            m_maturazione(Crit3DDate()),
            m_beta(beta),
            m_sigma(sigma),
            m_numeroPrimordi(4.),
            m_numeroFoglieEmerse(0.),
            m_limiteGradiGiorno(limiteGradiGiorno),
            m_sogliaGradiGiorno(9.),
            m_faseFenologica(std::vector<double>())
        {}

        ~Cereali()
        {
            m_faseFenologica.clear();
        }

        double TassoSviluppoEmergenza(const double& T) { return std::max( 0., -0.006 + 0.0065*T); }
        double NumeroFoglieEmerse(const double& P) { return ( 1. - exp( -0.03 * ( P - 4. ) ) ) / 0.03; }

        // calcolo delle date di semina ed emergenza fittizie
        bool CalcoloDateFittizie(const Stazione& stazione, const Parametri& parametri, Console& console);

        // calcolo del numero di foglie totali
        bool CalcoloNumeroFoglie(Stazione& stazione, const Parametri& parametri, Console& console);

        // calcolo della data di emergenza
        void Emergenza(const Stazione& stazione);

        // calcolo del numero di bozze fogliari e del periodo di viraggio apicale
        void Viraggio(const Stazione& stazione);

        // calcolo della data di spigatura
        void Spigatura(const Stazione& stazione);

        // calcolo della data di maturazione fisiologica
        void Maturazione(const Stazione& stazione);

        // simulazione dello sviluppo delle fasi fenologiche dei cereali
        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);

        // output del valore della fase fenologico
        double FaseFenologica(const unsigned long &giorno, const Parametri& parametri);
    };

#endif
