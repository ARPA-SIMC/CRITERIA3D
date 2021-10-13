
#include "girasole.h"

#include "math.h"
#include <algorithm>

void Girasole::Emergenza(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
	double Ts = 7.9;
    m_faseFenologica[i] += std::max( static_cast<double>(0.), Tm - Ts ) / 67.;
}

void Girasole::GemmaFiorale(const Stazione& stazione, const Parametri& parametri) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
	double Ts = 6.6;
    double dT = std::max( static_cast<double>(0.), Tm - Ts );

    m_faseFenologica[i] += ( C1[0][parametri.varieta] * dT - C2[0][parametri.varieta] * dT * dT )
		                 * ( C3[0][parametri.varieta] + C4[0][parametri.varieta] * Tm );
}

void Girasole::Fioritura(const Stazione& stazione, const Parametri& parametri) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
	double Ts = 3.9;
    double dT = std::max( static_cast<double>(0.), Tm - Ts );

    m_faseFenologica[i] += ( C1[1][parametri.varieta] * dT - C2[1][parametri.varieta] * dT * dT )
		                 * ( C3[1][parametri.varieta] + C4[1][parametri.varieta] * Tm );
}

void Girasole::Maturazione(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

	m_faseFenologica[i] += Tm / m_limiteGradiGiorno;
}

void Girasole::Raccolta(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

	m_faseFenologica[i] += Tm / m_limiteGradiGiorno;
}

void Girasole::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console) 
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
			Emergenza(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 2. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
				m_emergenza = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 2:
			GemmaFiorale(stazione, parametri);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 3.;
				m_gemmaFiorale = stazione.getDate(m_faseFenologica.size()-1);
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

		case 5:
			Raccolta(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 6. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 6.;
				m_raccolta = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		default:
			// nessuna azione;
			break;
	}
}

double Girasole::FaseFenologica(const unsigned long& giorno, const Parametri& parametri)
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
				bbch = 9. + ( 63. - 9. ) * ( m_faseFenologica[giorno] - 2. );
				break;

			case 3:
				bbch = 63. + ( 65. - 63. ) * ( m_faseFenologica[giorno] - 3. );
				break;

			case 4:
				bbch = 65. + ( 87. - 65. ) * ( m_faseFenologica[giorno] - 4. );
				break;

			case 5:
				bbch = 87. + ( 99. - 87. ) * ( m_faseFenologica[giorno] - 5. );
				break;

			case 6:
				bbch = 99.;
				break;

			default:
				// nessuna azione;
				bbch = parametri.dato_mancante;
		}
	}

	return bbch; 
}
