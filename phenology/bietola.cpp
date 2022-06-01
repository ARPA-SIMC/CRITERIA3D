#include <algorithm>
#include "bietola.h"
#include "math.h"


double Bietola::R(double T) 
{
  T = std::min(T, m_TEx);
  T = std::max(T, m_TEn);
  return m_Rmax * pow((T - m_TEn) / (m_TEo - m_TEn), m_alfa) 
	            * pow((m_TEx - T) / (m_TEx - m_TEo), m_alfa * ( m_TEx / m_TEo - 1.));
}


// Emergenza
void Bietola::Emergenza(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
	m_gradiGiornoEmergenza += Tm;
	m_faseFenologica[i] += R(Tm);
}


// 12 foglie
void Bietola::DodiciFoglie(const Stazione& stazione)
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_gradiGiornoDodiciFoglie += std::max(0., Tm);
    m_faseFenologica[i] += std::max(0., Tm) / ( 924. - m_gradiGiornoEmergenza );
}

// Raccolta
void Bietola::Raccolta(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;
    m_faseFenologica[i] += std::max(0., Tm) / ( m_limiteGradiGiorno - 924. );
}


void Bietola::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console) 
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
			DodiciFoglie(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 3.;
				m_dodiciFoglie = stazione.getDate(m_faseFenologica.size()-1);
			}
				
			break;

		case 3:
			Raccolta(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 4. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 4.;
				m_raccolta = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		default:
			// nessuna azione;
			break;
	}
}

double Bietola::FaseFenologica(const unsigned long& giorno, const Parametri& parametri)
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
				bbch = 9. + ( 19. - 9. ) * ( m_faseFenologica[giorno] - 2. );
				break;

			case 3:
				bbch = 19. + ( 99. - 19. ) * ( m_faseFenologica[giorno] - 3. );
				break;

			case 4:
				bbch = 99.;
				break;

			default:
				// nessuna azione;
				bbch = parametri.dato_mancante;
		}
	}

	return bbch; 
}

