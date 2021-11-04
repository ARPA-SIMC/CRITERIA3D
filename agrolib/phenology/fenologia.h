#ifndef FENOLOGIA_H
#define FENOLOGIA_H

    #include "coltura.h"
#ifndef _MAP_
    #include <map>
#endif


// LC decidere se eliminare le define e utilizzare enum
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

    enum phenoCrop {frumento_tenero, frumento_duro, orzo, mais, girasole, bietola, soia, pomodoro_seme, pomodoro_trapianto, olivo, vite, invalidCrop};
    enum phenoComputation {currentStage, anomalyDays, differenceStages};
    enum phenoScale {ARPA, BBCH};
    enum phenoVariety {precocissima, precoce, media, tardive};

    const std::map<phenoCrop, std::string> MapPhenoCropToString = {
        { frumento_tenero, "FRUMENTO_TENERO"} ,
        { frumento_duro, "FRUMENTO_DURO" },
        { orzo, "ORZO" },
        { mais, "MAIS" },
        { girasole, "GIRASOLE" },
        { bietola, "BIETOLA" },
        { soia, "SOIA" },
        { pomodoro_seme, "POMODORO_SEME" },
        { pomodoro_trapianto, "POMODORO_TRAPIANTO" },
        { olivo, "OLIVO" },
        { vite, "VITE" }
    };

    phenoCrop getKeyMapPhenoCrop(std::map<phenoCrop, std::string> map, const std::string& value);
    std::string getStringMapPhenoCrop(std::map<phenoCrop, std::string> map, phenoCrop crop);
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
