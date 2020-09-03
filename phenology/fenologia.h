#ifndef FENOLOGIA_H
#define FENOLOGIA_H

    #include "coltura.h"

    #define FRUMENTO_TENERO 0
    #define FRUMENTO_DURO 1
    #define ORZO 2
    #define MAIS 3
    #define GIRASOLE 4
    #define BIETOLA 5
    #define SOIA 6
    #define POMODORO_SEME 7
    #define POMODORO_TRAPIANTO 8
    #define OLIVO 9
    #define VITE 10


    // chiamata principale
    void feno( const float soglia, const int coltura, const int varieta, const char* logfile,
           const int max_giorni_interpolazione, const float dato_mancante, const int tipoScala,
           const int giornoInizio, const int meseInizio, const int annoInizio, const int numeroGiorni,
           char* nome, char* codice, const int lat_gradi, const int lat_primi, const float lat_secondi,
           const int quota, float *tmin, float *tmax, float *prec, float* valori );


    class Fenologia
    {
        Coltura* m_coltura;

    public:
        Fenologia() : m_coltura(nullptr) {}
        Fenologia(const Coltura& coltura) : m_coltura(new Coltura(coltura)) {}

        ~Fenologia()
        { delete m_coltura; }

        bool SceltaColtura(const Parametri& parametri, Console& console);
        void CalcolaFase(Stazione& stazione, const Parametri& parametri, Console& console);
        double FaseFenologica(const unsigned long& giorno, const Parametri& parametri);
    };


#endif
