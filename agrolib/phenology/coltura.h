#ifndef COLTURA_H
#define COLTURA_H

    #include "console.h"
    #include "stazione.h"

    // classe astratta per tutte le colture
    class Coltura
    {
    public:
        virtual void Fenologia(Stazione&, const Parametri&, Console&) {}
        virtual double FaseFenologica(const unsigned long&, const Parametri&) { return 0; }
     };

#endif
