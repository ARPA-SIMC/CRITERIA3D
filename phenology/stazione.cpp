#include "stazione.h"
#include "commonConstants.h"

#include "math.h"


// calcolo del fotoperiodo alla data corrente
double Stazione::Fotoperiodo(const long& i) 
{
  double theta = -(23.45 * DEG_TO_RAD) * cos(2. * PI * (getDoyFromDate(m_data[unsigned(i)]) + 10. ) / 365.);
  double lat = m_latitudine.Gradi() + m_latitudine.Primi() / 60. + m_latitudine.Secondi() / 3600.;
  lat *= DEG_TO_RAD;
  double fotoPeriodo = 12. * (1. + 2.*asin(-sin(-4. * DEG_TO_RAD) + sin(theta)*sin(lat) / cos(theta)/cos(lat)) / PI );

  return(fotoPeriodo);
}

void Stazione::SetAnagrafica(char* nome, char* codice, const long& quota, const Coordinata& latitudine)
{
	m_nome = nome;
	m_codice = codice;
	m_quota = quota;
	m_latitudine = latitudine;
}

void Stazione::SetDati(float* tmin, float* tmax, float* prec, const Parametri& parametri)
{
	// index_giorno andrebbe controllato
	for( long i = 0; i < parametri.numeroGiorni; i++)
	{
		if( m_data.empty() )
			m_data.push_back(parametri.dataInizio);
		else
		{
			m_data.push_back(m_data[m_data.size()-1]);
			++m_data[m_data.size()-1];
		}

        m_tmin.push_back(tmin[i]);
        m_tmax.push_back(tmax[i]);
		m_prec.push_back(prec[i]);
	}
}


// interpola i dati per evitare buchi - solo temperature
bool Stazione::InterpolaDati(const Parametri& parametri, Console& console)
{	
	char message[256];

	// Temperatura minime
	if( interpolaDatiTemperatura(m_nome, m_tmin, parametri, console) == false )
	{
		sprintf(message, "\n\n Stazione di %s", m_nome);
		console.Show(message);
		sprintf(message, "\n Dati finali di temperatura massima mancanti - non e' possibile interpolare i dati.");
		console.Show(message);
		return false;
	
	}
		
	// Temperatura massime
	if( interpolaDatiTemperatura(m_nome, m_tmax, parametri, console) == false )
	{
		sprintf(message, "\n\n Stazione di %s", m_nome);
		console.Show(message);
		sprintf(message, "\n Dati finali di temperatura massima mancanti - non e' possibile interpolare i dati.");
		console.Show(message);
		return false;
	
	}

	sprintf(message, "\n\nStazione di %s ------- Fine interpolazione.\n\n", m_nome);
	console.Show(message);

	return true;
}


bool Stazione::interpolaDatiTemperatura(char* nome, std::vector<float> &dati, const Parametri& parametri, Console& console)
{
    char message[256];

    unsigned long i;
    unsigned long first = dati.size();

    for( i = 0; i < dati.size(); i++)
    {
        if( dati[i] == parametri.dato_mancante )
        {
            first = i;
            break;
        }
    }

    if( first > parametri.max_giorni_interpolazione )
    {
        sprintf(message, "\n\n Stazione di %s", nome);
        console.Show(message);
        sprintf(message, "\n Troppi dati iniziali mancanti - impossibile interpolare i dati.");
        console.Show(message);
        return false;
    }

    // replica in tutti i giorni del buco iniziale il primo valore valido
    if( first > 0 )
    {
        sprintf(message, "\n\n Stazione di %s - aggiunte temperature minime iniziali", nome);
        for( i = 0; i < first; i++ )
            dati[i] = dati[first];
    }

    long missing = 0;

    for( i = first + 1; i < dati.size(); i++ )
    {
        if( dati[i] == parametri.dato_mancante )
        {
            missing++;
            continue;
        }

        if( missing == 0)
        {
            first = i;
            continue;
        }

        if( missing > parametri.max_giorni_interpolazione )
        {
            sprintf(message, "\n\n Stazione di %s", nome);
            console.Show(message);
            sprintf(message, "\n Troppi dati mancanti - non e' possibile interpolare i dati.");
            console.Show(message);
            return false;
        }
        else
        {
            double step = (dati[i] - dati[first]) / static_cast<double>(i - first);

            for( long j = first + 1; j < i; j++ )
                dati[j] = dati[first] + step * (j - first);

            missing = 0;
        }
    }

    // mancano ultimi valori
    if( missing > 0 )
    {
        sprintf(message, "\n\n Stazione di %s", nome);
        console.Show(message);
        sprintf(message, "\nDati finali mancanti - non e' possibile interpolare i dati.");
        console.Show(message);
        return false;
    }

    return true;
}
