
#include "vite.h"

#include "math.h"
#include <algorithm>

double Vite::GradiGiorno(const long& i, const Stazione& stazione)
{
	double Ts = 10.;

	double gradiGiorno = 0.;

    if( stazione.Tn(i) < Ts && stazione.Tx(i) > Ts )
        gradiGiorno += std::max( static_cast<double>(0.),
							 .5 * ( stazione.Tx(i) - Ts ) * ( stazione.Tx(i) - Ts ) /
							 ( stazione.Tx(i) - stazione.Tn(i) ) );
    else
        gradiGiorno += std::max( static_cast<double>(0.),
							  .5 * ( stazione.Tx(i) + stazione.Tn(i) ) - Ts );

	return gradiGiorno;
}


void Vite::Germogliamento(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += GradiGiorno(i, stazione) / m_limiteGradiGiorno[0];
}

void Vite::Fioritura(const Stazione& stazione)
{
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += GradiGiorno(i, stazione) / ( m_limiteGradiGiorno[1] - m_limiteGradiGiorno[0] );
}

void Vite::Invaiatura(const Stazione& stazione)
{
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += GradiGiorno(i, stazione) / ( m_limiteGradiGiorno[2] - m_limiteGradiGiorno[1] );
}

void Vite::Maturazione(const Stazione& stazione)
{
	long i = m_faseFenologica.size() - 1;
    m_faseFenologica[i] += GradiGiorno(i, stazione) / ( m_limiteGradiGiorno[3] - m_limiteGradiGiorno[2] );
}


void Vite::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console) 
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
				m_inizioAttivita = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 1:
			Germogliamento(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 2. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
				m_germogliamento = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		case 2:
			Fioritura(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 3. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 3.;
				m_fioritura = stazione.getDate(m_faseFenologica.size()-1);
			}
				
			break;

		case 3:
			Invaiatura(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 4. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 4.;
				m_invaiatura = stazione.getDate(m_faseFenologica.size()-1);
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

double Vite::FaseFenologica(const unsigned long& giorno, const Parametri& parametri) 
{ 
	double bbch = m_faseFenologica[giorno];

	if( parametri.scalaBBCH )
	{
		switch( static_cast<long>( floor(m_faseFenologica[giorno]) ) )
		{
			case 1:
				bbch = 8. * ( m_faseFenologica[giorno] - 1. );
				break;

			case 2:
				bbch = 8. + ( 61. - 8. ) * ( m_faseFenologica[giorno] - 2. );
				break;

			case 3:
				bbch = 61. + ( 83. - 61. ) * ( m_faseFenologica[giorno] - 3. );
				break;

			case 4:
				bbch = 83. + ( 87. - 83. ) * ( m_faseFenologica[giorno] - 4. );
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

