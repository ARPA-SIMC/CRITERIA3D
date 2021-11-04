
#include "olivo.h"

#include "math.h"
#include <algorithm>


void Olivo::InduzioneFiorale(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

	double numeroGiorni = 0.; 

    if ( Tm >= 12.2 && Tm <= 13.3 )
		numeroGiorni += 1.;
    else if ( Tm >= 9.7 && Tm < 12.2 && stazione.Tx(i) >= 12.5 )
		numeroGiorni += ( Tm - 9.7 ) / 2.5;
    else if ( Tm > 13.3 && Tm <= 15.8 && stazione.Tx(i) <= 21.1 )
		numeroGiorni += ( Tm - 13.3 ) / 2.5;
    
	m_faseFenologica[i] += numeroGiorni / 17.;
}

void Olivo::Fioritura(const Stazione& stazione)
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

	double Ts = 12.5;
	double gradiGiorno = 0.;

	if( Tm > 12.5 )
        gradiGiorno += std::max( static_cast<double>(0.), Tm - Ts );
    else if( stazione.Tx(i) > 12.5 )
        gradiGiorno += std::max( static_cast<double>(0.),
		                      ( stazione.Tx(i) - 12.5 ) *( stazione.Tx(i) - 12.5 ) / 2. /
							  ( stazione.Tx(i) - stazione.Tn(i) ) );
  
	m_faseFenologica[i] += gradiGiorno / m_limiteGradiGiorno[0];
}

void Olivo::Maturazione(const Stazione& stazione) 
{
	long i = m_faseFenologica.size() - 1;
	double Tm = ( stazione.Tn(i) + stazione.Tx(i) ) / 2.;

	double Ts = 12.5;
	double gradiGiorno = 0.;

	if( Tm > 12.5 )
        gradiGiorno += std::max( static_cast<double>(0.), Tm - Ts );
    else if( stazione.Tx(i) > 12.5 )
        gradiGiorno += std::max( static_cast<double>(0.),
		                      ( stazione.Tx(i) - 12.5 ) *( stazione.Tx(i) - 12.5 ) / 2. /
							  ( stazione.Tx(i) - stazione.Tn(i) ) );
  
	m_faseFenologica[i] += gradiGiorno / m_limiteGradiGiorno[1];
}


void Olivo::Fenologia(Stazione& stazione, const Parametri& parametri, Console& console) 
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
			InduzioneFiorale(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 2. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 2.;
				m_induzioneFiorale = stazione.getDate(m_faseFenologica.size()-1);
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
			Maturazione(stazione);

			if( m_faseFenologica[m_faseFenologica.size()-1] >= 4. )
			{
				m_faseFenologica[m_faseFenologica.size()-1] = 4.;
				m_maturazione = stazione.getDate(m_faseFenologica.size()-1);
			}

			break;

		default:
			// nessuna azione;
			break;
	}
}

double Olivo::FaseFenologica(const unsigned long& giorno, const Parametri& parametri) 
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
				bbch = 9. + ( 65. - 9. ) * ( m_faseFenologica[giorno] - 2. );
				break;

			case 3:
				bbch = 65. + ( 89. - 65. ) * ( m_faseFenologica[giorno] - 3. );
				break;

			case 4:
				bbch = 89.;
				break;

			default:
				// nessuna azione;
				bbch = parametri.dato_mancante;
		}
	}

	return bbch; 
}

