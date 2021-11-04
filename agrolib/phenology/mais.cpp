
#include "mais.h"

#include "math.h"
#include <algorithm>

extern long doy(const Crit3DDate& data);

// funzione semplificata per il calcolo della temperatura del suolo
double Mais::TemperaturaSuoloMinima(const long& giorno, const Stazione& stazione, const Parametri& parametri)
{
	// frazione dei giorni piovosi (wet days?)
	double Pw[12] =  { 0.238, 0.253, 0.245, 0.25,  0.242, 0.216, 0.145, 0.171, 0.176, 0.251, 0.323, 0.251 };

	double albedo = 0.2;	// albedo 
	double lag = 0.5;		// coefficiente di ritardo 
	double fz = 0.95;

	double  pi = 4 * asin(1.);
    double Ra = 18. + 12. * sin( ( getDoyFromDate(stazione.getDate(giorno)) - 100. ) / 365. * 2. * pi);
	double Rah = 2. * ( (Ra * (1. - albedo) * 0.75 / 15 ) * ( Ra * (1. - albedo) * 0.75 / 15 ) - 1. );

	double Tg;

	if( stazione.Pr(giorno) != parametri.dato_mancante && stazione.Pr(giorno) > 0. )
		Tg = stazione.Tn(giorno) + 0.1 * (stazione.Tx(giorno) - stazione.Tn(giorno)) + Rah;
	else
        Tg = ( (stazione.Tx(giorno) + stazione.Tn(giorno)) / 2. - Pw[stazione.getDate(giorno).month]
        * (stazione.Tn(giorno) + 0.1 * (stazione.Tx(giorno) - stazione.Tn(giorno))) )
        / ( 1. - Pw[stazione.getDate(giorno).month] ) + Rah;

	double Tsn;

	if( giorno == 0 )
		Tsn = lag * stazione.Tn(giorno) + (1. - lag) * ( fz * (stazione.Tyear() - Tg) + Tg );
	else
		Tsn = lag * stazione.Tn(giorno - 1) + (1. - lag) * ( fz * (stazione.Tyear() - Tg) + Tg );

	return Tsn;
}

double Mais::TemperaturaSuoloMassima(const long& giorno, const Stazione& stazione, const Parametri& parametri)
{
	// frazione dei giorni piovosi (wet days?)
	double Pw[12] =  { 0.238, 0.253, 0.245, 0.25,  0.242, 0.216, 0.145, 0.171, 0.176, 0.251, 0.323, 0.251 };

	double albedo = 0.2;	// albedo 
	double lag = 0.5;		// coefficiente di ritardo 
	double fz = 0.95;

	double  pi = 4 * asin(1.);
    double Ra = 18. + 12. * sin( ( getDoyFromDate(stazione.getDate(giorno)) - 100. ) / 365. * 2. * pi);
	double Rah = 2. * ( (Ra * (1. - albedo) * 0.75 / 15 ) * ( Ra * (1. - albedo) * 0.75 / 15 ) - 1. );

	double Tg;

	if( stazione.Pr(giorno) != parametri.dato_mancante && stazione.Pr(giorno) > 0. )
		Tg = stazione.Tn(giorno) + 0.1 * (stazione.Tx(giorno) - stazione.Tn(giorno)) + Rah;
	else
        Tg = ( (stazione.Tx(giorno) + stazione.Tn(giorno)) / 2. 
        - Pw[stazione.getDate(giorno).month] * ( stazione.Tn(giorno) + 0.1 * (stazione.Tx(giorno) - stazione.Tn(giorno)) ) )
        / ( 1. - Pw[stazione.getDate(giorno).month] ) + Rah;

	double Tsx;

	if( giorno == 0 )
		Tsx = lag * stazione.Tx(giorno) + (1. - lag) * ( fz * (stazione.Tyear() - Tg) + Tg );
	else
		Tsx = lag * stazione.Tx(giorno - 1) + (1. - lag) * ( fz * (stazione.Tyear() - Tg) + Tg );	

	return Tsx;
}

// Calcolo del numero delle foglie
void Mais::CalcoloNumeroFoglie(const long& i, Stazione& stazione, const Parametri& parametri)
{
    long numeroGiorni = getDoyFromDate(stazione.getDate(i)) - getDoyFromDate(m_emergenza);
	double Tsn = TemperaturaSuoloMinima(i, stazione, parametri);
	double Tsx = TemperaturaSuoloMassima(i, stazione, parametri);
	m_numeroFoglieTotali = ( numeroGiorni - 1) * m_numeroFoglieTotali + NumeroFoglieMinimo() + 
							static_cast<long>( 0.1 * ( NumeroFoglieMinimo() - 10 ) * ( stazione.Fotoperiodo(i) - 12.5 ) +
							0.5 * ( EffettoTemperatura(Tsn) + EffettoTemperatura(Tsx) ) );
	m_numeroFoglieTotali /= numeroGiorni;
}

// calcolo della data di emergenza
void Mais::Emergenza(const Stazione& stazione, const Parametri& parametri)
{
	long i = m_faseFenologica.size() - 1;
	double Tsm = ( TemperaturaSuoloMinima(i, stazione, parametri) + TemperaturaSuoloMassima(i, stazione, parametri) ) / 2.;
    m_faseFenologica[i] += std::max( static_cast<double>(0.), 0.0144 * Tsm - 0.09193 );
}

// routine di calcolo della data di viraggio apicale
void Mais::Viraggio(Stazione& stazione, const Parametri& parametri) 
{
	long i = m_faseFenologica.size() - 1;

	// calcolo delle foglie totali al giorno corrente
	CalcoloNumeroFoglie(i, stazione, parametri);
	double Tsn = TemperaturaSuoloMinima(i, stazione, parametri);
	double Tsx = TemperaturaSuoloMassima(i, stazione, parametri);

	// determinazione delle foglie emerse al momento del viraggio apicale
	m_numeroFoglieEmerse += .5 * ( TassoCrescitaFoglie(Tsn) + TassoCrescitaFoglie(Tsx) );
	m_faseFenologica[i] += .5 * ( TassoCrescitaPrimordi(Tsn) + TassoCrescitaPrimordi(Tsx) ) / ( m_numeroFoglieTotali - 5. );
}

// routine di calcolo della data di m_fioritura
void Mais::Fioritura(const Stazione& stazione, const Parametri& parametri)
{
	long i = m_faseFenologica.size() - 1;
	m_faseFenologica[i] += .5 * ( TassoCrescitaFoglie(stazione.Tn(i)) + TassoCrescitaFoglie(stazione.Tx(i)) ) / ( m_numeroFoglieTotali +1. - m_numeroFoglieEmerse );
}

// routine per il calcolo della data di m_maturazione fisiologica del mais
void Mais::Maturazione(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += CornHeatUnits(std::max(10., (double)stazione.Tx(i)), std::max(4.4, (double)stazione.Tn(i))) / m_limiteGradiGiorno[m_classeFAO/100 - 2];
}

// simulazione dello sviluppo delle fasi fenologiche del mais
void Mais::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console) 
{
	char message[256];

	if( m_faseFenologica.empty() )
	{
		m_faseFenologica.push_back( static_cast<double>(0.) );
	}
	else
	{
		if( m_faseFenologica[m_faseFenologica.size()-1] == parametri.dato_mancante )
		{
			m_faseFenologica[m_faseFenologica.size()-1] = parametri.dato_mancante;
			sprintf(message, "Exception raised: dati fenologici mancanti\n");
			console.Show(message);
			sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
			console.Show(message);
			return;
		}
			
		m_faseFenologica.push_back( m_faseFenologica[m_faseFenologica.size()-1] );
	}

	if( stazione.Tn(m_faseFenologica.size()-1) == parametri.dato_mancante || 
		stazione.Tx(m_faseFenologica.size()-1) == parametri.dato_mancante )
	{
		sprintf(message, "Exception raised: dati di temperatura mancanti\n");
		console.Show(message);
		sprintf(message, "Coltura %s\tStazione di %s", m_coltura, stazione.Nome());
		console.Show(message);
		return;
	}

	switch( static_cast<long>( floor(m_faseFenologica[m_faseFenologica.size()-1]) ) )
	{
		case 0:
			if( stazione.getDate(m_faseFenologica.size()-1) == parametri.dataInizio )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 1.;
				m_semina = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 1:
			Emergenza(stazione, parametri);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 2. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
				m_emergenza = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 2:
			Viraggio(stazione, parametri);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 3.;
				m_viraggioApicale = stazione.getDate(m_faseFenologica.size()-1);
			}
				
			break;

		case 3:
			Fioritura(stazione, parametri);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 4.)
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 4.;
				m_fioritura = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 4:
			Maturazione(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 5. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 5.;
				m_maturazione = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		default:
			// nessuna azione;
			break;
	}
}

double Mais::FaseFenologica(const unsigned long& giorno, const Parametri& parametri)
{ 
	double bbch = m_faseFenologica[giorno];

	if( parametri.scalaBBCH )
	{
		switch( static_cast<long>( floor(m_faseFenologica[giorno]) ) )
		{
			case 1:
				bbch = 9. * ( m_faseFenologica[giorno] - 1. );
				break;

			case 2:
				bbch = 9. + ( 17. - 9. ) * ( m_faseFenologica[giorno] - 2. );
				break;

			case 3:
				bbch = 17. + ( 53. - 17. ) * ( m_faseFenologica[giorno] - 3. );
				break;

			case 4:
				bbch = 53. + ( 87. - 53. ) * ( m_faseFenologica[giorno] - 4. );
				break;

			case 5:
				bbch = 87.;
				break;

			default:
				// nessuna azione;
				bbch = parametri.dato_mancante;
		}
	}

	return bbch; 
}


