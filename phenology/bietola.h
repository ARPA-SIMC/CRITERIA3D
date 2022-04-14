#ifndef BIETOLA_H
#define BIETOLA_H

    #include "coltura.h"

    class Bietola : public Coltura
    {
        const char* m_coltura;

        Crit3DDate m_semina;
        Crit3DDate m_emergenza;
        Crit3DDate m_dodiciFoglie;
        Crit3DDate m_raccolta;

        double m_limiteGradiGiorno;

        double m_gradiGiornoEmergenza;
        double m_gradiGiornoDodiciFoglie;

        double m_Rmax;		// valore massimo di radiazione
        double m_TEn;		// temperatura minima perchè la radiazione abbia effetto attivo
        double m_TEx;		// temperatura massima perchè la radiazione abbia effetto attivo
        double m_TEo;		// temperatura ottimale per l'effetto radiativo
        double m_alfa;		// coefficiente dell'esponente

        std::vector<double> m_faseFenologica;

    public:
        Bietola()
        :	m_coltura(nullptr),
            m_semina(Crit3DDate()),
            m_dodiciFoglie(Crit3DDate()),
            m_raccolta(Crit3DDate()),
            m_limiteGradiGiorno(0.),
            m_gradiGiornoEmergenza(0.),
            m_gradiGiornoDodiciFoglie(0.),
            m_Rmax(0.),
            m_TEn(0.),
            m_TEx(0.),
            m_TEo(0.),
            m_alfa(0.),
            m_faseFenologica(std::vector<double>())
        {}

        Bietola(const char* coltura, double limiteGradiGiorno)
        :	m_coltura(coltura),
            m_semina(Crit3DDate()),
            m_dodiciFoglie(Crit3DDate()),
            m_raccolta(Crit3DDate()),
            m_limiteGradiGiorno(limiteGradiGiorno),
            m_gradiGiornoEmergenza(0.),
            m_gradiGiornoDodiciFoglie(0.),
            m_Rmax(0.3),
            m_TEn(2.),
            m_TEx(36.),
            m_TEo(27.3),
            m_alfa(2.),
            m_faseFenologica(std::vector<double>())
        {}

        ~Bietola()
        {
            m_faseFenologica.clear();
        }

        double R(double T);

        // other functions
        void Emergenza(const Stazione& stazione);
        void DodiciFoglie(const Stazione& stazione);
        void Raccolta(const Stazione& stazione);

        void Fenologia(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long &giorno, const Parametri& parametri);
    };

#endif


