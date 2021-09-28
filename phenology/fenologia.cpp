/*  fenologia.cpp
    Libreria per la simulazione dello sviluppo di alcune colture:
    frumento, grano duro, mais, orzo, girasole, soia, bietola e pomodoro
    seguendo gli algoritmi contenuti nel rapporto
    di F. Nerozzi, F. Zinoni e V. Marletto (rapporto tecnico ARPA-SMR, 1996)

    1996	- realizzato da F. Nerozzi
    2005    - modificato da F. Tomei per essere chiamata da PRAGA
    2011	- modificato da F. Nerozzi:
                - riscrittura delle classi (solo per cereali (frumento tenero, duro ed orzo) e mais)
                - definizione di un unico entry point
                - conversione dell'uscita dei modelli fenologici in scala BBCH (indicazioni di Giulia Villani)
    2019    - modificato da F. Tomei: uniformato alle altre librerie di CRITERIA-3D distribution
*/

#include "fenologia.h"
#include "cereali.h"
#include "mais.h"
#include "girasole.h"
#include "bietola.h"
#include "soia.h"
#include "pomodoro.h"
#include "olivo.h"
#include "vite.h"

#include "math.h"

phenoCrop getKeyMapPhenoCrop(std::map<phenoCrop, std::string> map, const std::string& value)
{
    std::map<phenoCrop, std::string>::const_iterator it;
    phenoCrop key = invalidCrop;

    for (it = map.begin(); it != map.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}

std::string getStringMapPhenoCrop(std::map<phenoCrop, std::string> map, phenoCrop crop)
{
    auto search = map.find(crop);

    if (search != map.end())
        return search->second;

    return "";
}

void feno( float soglia, int coltura, int varieta, char* logfile, int max_giorni_interpolazione,
           float dato_mancante, int tipoScala, int giornoInizio, int meseInizio, int annoInizio,
           int numeroGiorni, char* nome, char* codice, int lat_gradi, int lat_primi, float lat_secondi,
           int quota, float *tmin, float *tmax, float *prec, float* valori )
{
    Console console;
    login(logfile, console);

    // assegna parametri
    Parametri parametri;
    parametri.sogliaVernalizzazione = soglia;
    parametri.coltura = coltura;
    parametri.varieta = varieta;
    parametri.dato_mancante = dato_mancante;
    parametri.scalaBBCH = tipoScala;   // 0 uscita diretta del modello; 1 scala BBCH
    parametri.max_giorni_interpolazione = max_giorni_interpolazione;
    parametri.dataInizio = Crit3DDate(giornoInizio, meseInizio, annoInizio);
    parametri.numeroGiorni = numeroGiorni;

    // assegna dati meteo
    Stazione stazione;
    stazione.SetAnagrafica(nome, codice, quota, Coordinata(lat_gradi, lat_primi, lat_secondi));
    stazione.SetDati(tmin, tmax, prec, parametri);

    // calcola fase fenologica
    Fenologia fenologia;

    if( fenologia.SceltaColtura(parametri, console) )
    {
        fenologia.CalcolaFase(stazione, parametri, console);

        for( long giorno = 0; giorno < stazione.NumeroGiorni(); giorno++ )
            valori[giorno] = static_cast<float>(fenologia.FaseFenologica(giorno, parametri));
    }
    else
    {
        for( long giorno = 0; giorno < stazione.NumeroGiorni(); giorno++ )
            valori[giorno] = static_cast<float>(parametri.dato_mancante);
    }

    logout(console);
}


bool Fenologia::SceltaColtura(const Parametri& parametri, Console& console)
{
	char message[256];

	bool success = false;

    if( parametri.coltura == FRUMENTO_TENERO )
	{
		switch( parametri.varieta )
		{
			case 0:
                m_coltura = new Cereali("Frumento (precoci)", 0.0149, 24.5, 400.);
				break;

			case 1:
				m_coltura = new Cereali("Frumento (medie)", 0.0149, 32.1, 425.);
				break;

			case 2:
				m_coltura = new Cereali("Frumento (tardive)", 0.0149, 40.1, 450.);
				break;

			default:
				m_coltura = new Cereali("Frumento (medie)", 0.0149, 32.1, 425.);
				break;
		}

		success = true;
	}
    else if( parametri.coltura == FRUMENTO_DURO )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Cereali("Frumento duro (precoci)", 0.0149, 40, 400.);
				break;

			case 1:
				m_coltura = new Cereali("Frumento duro (medie)", 0.0149, 45, 425.);
				break;

			case 2:
				m_coltura = new Cereali("Frumento duro (tardive)", 0.0149, 50, 450.);
				break;

			default:
				m_coltura = new Cereali("Frumento duro (medie)", 0.0149, 45, 425.);
				break;
		}

		success = true;
	}
    else if( parametri.coltura == ORZO )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Cereali("Orzo (precoci)", 0.0178, 20, 300.);
				break;

			case 1:
				m_coltura = new Cereali("Orzo (medie)", 0.0178, 25, 325.);
				break;

			case 2:
				m_coltura = new Cereali("Orzo (tardive)", 0.0178, 30, 350.);
				break;

			default:
				m_coltura = new Cereali("Orzo (medie)", 0.0178, 25, 325.);
				break;
		}

		success = true;
	}

    else if( parametri.coltura == MAIS )
	{
        m_coltura = new Mais("Mais", parametri.varieta);

		success = true;
	}

    else if( parametri.coltura == GIRASOLE )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Girasole("Girasole (precocissime)", 553. );
				break;

			case 1:
				m_coltura = new Girasole("Girasole (precoci)", 603.);
				break;

			case 2:
				m_coltura = new Girasole("Girasole (medie)", 650.);
				break;

			case 3:
				m_coltura = new Girasole("Girasole (tardive)", 745.);
				break;

			default:
				m_coltura = new Girasole("Girasole (medie)", 650.);
		}

		success = true;
	}
    else if( parametri.coltura == BIETOLA )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Bietola("Bietola (precoci)", 2750.);
				break;

			case 1:
				m_coltura = new Bietola("Bietola (medie)", 3100.);
				break;

			case 2:
				m_coltura = new Bietola("Bietola (tardive)", 3450.);
				break;

			default:
				m_coltura = new Bietola("Bietola (medie)", 3100.);
				break;
		}

		success = true;
	}
    else if( parametri.coltura == SOIA )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Soia("Soia (classe 0)");
				break;

			case 1:
				m_coltura = new Soia("Soia (classe 1)");
				break;

			case 2:
				m_coltura = new Soia("Soia (classe 2)");
				break;

			default:
				m_coltura = new Soia("Soia (classe 1)");
				break;
		}

		success = true;
	}
    else if( parametri.coltura == POMODORO_SEME )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Pomodoro("Pomodoro da semina (precoci)", 3., 7.);
				break;

			case 1:
				m_coltura = new Pomodoro("Pomodoro da semina (medie)", 5., 9.);
				break;

			case 2:
				m_coltura = new Pomodoro("Pomodoro da semina (tardive)", 7., 11.);
				break;

			default:
				m_coltura = new Pomodoro("Pomodoro da semina (medie)", 5., 9.);
				break;
		}

		success = true;
	}
    else if( parametri.coltura == POMODORO_TRAPIANTO )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new PomodoroTrapianto("Pomodoro da trapianto (precoci)", 3., 7.);
				break;

			case 1:
				m_coltura = new PomodoroTrapianto("Pomodoro da trapianto (medie)", 5., 9.);
				break;

			case 2:
				m_coltura = new PomodoroTrapianto("Pomodoro da trapianto (tardive)", 7., 11.);
				break;

			default:
				m_coltura = new PomodoroTrapianto("Pomodoro da trapianto (medie)", 5., 9.);
				break;
		}

		success = true;
	}
    else if(parametri.coltura == OLIVO )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Olivo("Olivo (precoci)", 200., 1325.);
				break;

			case 1:
				m_coltura = new Olivo("Olivo (medie)", 215., 1340.);
				break;

			case 2:
				m_coltura = new Olivo("Olivo (tardive)", 230., 1355.);
				break;

			default:
				m_coltura = new Olivo("Olivo (medie)", 215., 1340.);
				break;
		}

		success = true;
	}
    else if( parametri.coltura == VITE )
	{
		switch( parametri.varieta )
		{
			case 0:
				m_coltura = new Vite("Vite (precocissime)", 104., 388., 988., 1300.);
				break;

			case 1:
				m_coltura = new Vite("Vite (precoci)", 106., 420., 1040., 1500.);
				break;

			case 2:
				m_coltura = new Vite("Vite (medie)", 109., 420., 1120., 1750.);
				break;

			case 3:
				m_coltura = new Vite("Vite (tardive)", 113., 441., 1241., 2000.);
				break;

			default:
				m_coltura = new Vite("Vite (medie)", 109., 420., 1120., 1750.);
				break;
		}

		success = true;
	}
	else
	{
		sprintf(message, "Exception raised: indice della coltura non definito\n");
		console.Show(message);

		success = false;
	}

	return success;
}

double Fenologia::FaseFenologica(const unsigned long& giorno, const Parametri& parametri)
{
	return m_coltura->FaseFenologica(giorno, parametri);
}

void Fenologia::CalcolaFase(Stazione& stazione, const Parametri& parametri, Console& console)
{
	for( long i = 0; i < stazione.NumeroGiorni(); i++ ) 
		m_coltura->Fenologia(stazione, parametri, console);
}
