#ifndef STAZIONE_H
#define STAZIONE_H

    #include <vector>
    #include "crit3dDate.h"
    #include "console.h"

    typedef struct
    {
        int coltura;
        int varieta;
        int max_giorni_interpolazione;
        int numeroGiorni;
        int scalaBBCH;
        float dato_mancante;
        float sogliaVernalizzazione;
        Crit3DDate dataInizio;
    } Parametri;


    class Coordinata {
    private:
        long m_gradi;
        long m_primi;
        double m_secondi;

    public:
        Coordinata() : m_gradi(0), m_primi(0), m_secondi(0) {}

        Coordinata(const long& gradi, const long& primi, const double& secondi)
            : m_gradi(gradi), m_primi(primi), m_secondi(secondi) {}

        long Gradi() const { return m_gradi; }
        long Primi() const { return m_primi; }
        double Secondi() const { return m_secondi; }
    };


    class Stazione {
        char* m_nome;
        char* m_codice;
        long m_quota;
        Coordinata m_latitudine;
        Coordinata m_longitudine;
        std::vector<Crit3DDate> m_data;
        std::vector<float> m_tmin;
        std::vector<float> m_tmax;
        std::vector<float> m_prec;

    public:
        Stazione()
            :	m_nome(nullptr),
                m_codice(nullptr),
                m_quota(0),
                m_latitudine(Coordinata()),
                m_longitudine(Coordinata()),
                m_data(std::vector<Crit3DDate>()),
                m_tmin(std::vector<float>()),
                m_tmax(std::vector<float>()),
                m_prec(std::vector<float>())
        {}

        Stazione(char* nome, char* codice, const long& quota,
                 const Coordinata& latitudine, const Coordinata& longitudine,
                 std::vector<Crit3DDate>& data, std::vector<float>& tmin, std::vector<float>& tmax,
                 std::vector<float>& prec)
            :	m_nome(nome),
                m_codice(codice),
                m_quota(quota),
                m_latitudine(latitudine),
                m_longitudine(longitudine),
                m_data(data),
                m_tmin(tmin),
                m_tmax(tmax),
                m_prec(prec)
        {}

        ~Stazione()
        {
            m_tmin.clear();
            m_tmax.clear();
            m_prec.clear();
        }

        // inline functions
        char* Nome() const { return m_nome; }
        char* Codice() const { return m_codice; }
        long Quota() const { return m_quota; }
        long NumeroGiorni() const { return long(m_data.size()); }

        Crit3DDate getDate(const long& i) const { return m_data[unsigned(i)]; }
        float Tn(const long& i) const { return m_tmin[unsigned(i)]; }
        float Tx(const long& i) const { return m_tmax[unsigned(i)]; }
        float Pr(const long& i) const { return m_prec[unsigned(i)]; }
        float Tyear() const { return 12.6f; }	// media climatologica Emilia-Romagna

        // other functions
        double Fotoperiodo(const long& i);
        void SetAnagrafica(char* nome, char* codice, const long& quota, const Coordinata& latitudine);
        void SetDati(float* tmin, float* tmax, float* prec, const Parametri& parametri);
        bool InterpolaDati(const Parametri& parametri, Console& console);
        bool interpolaDatiTemperatura(char* nome, std::vector<float>& dati, const Parametri& parametri, Console& console);

    };

#endif
