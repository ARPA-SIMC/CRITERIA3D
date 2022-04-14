#include <time.h>
#include "console.h"
#include "string.h"


static clock_t start;


void login(const char* log, Console& console)
{
	char message[256];

	// start counting CPU time
	start = clock();

	// Get time as long integer
	time_t long_time;
	time( &long_time );
	
	// Convert to local time
	struct tm *newtime;
	newtime = localtime( &long_time );	

	// Set up extension
	char am_pm[] = "AM";
	if( newtime->tm_hour > 12 )			
		strcpy( am_pm, "PM" );

	// Convert from 24-hour to 12-hour clock
	if( newtime->tm_hour > 12 )			
		newtime->tm_hour -= 12;

	// Set hour to 12 if midnight
	if( newtime->tm_hour == 0 )			
		newtime->tm_hour = 12;

	console.Open(log, "w");

	sprintf(message,"Feno - Log File %.19s %s\n", asctime( newtime ), am_pm );
	console.Show(message);
	sprintf(message,"Inizio elaborazione\n");
	console.Show(message);
}


void logout(Console& console)
{
	char message[256];

	// stop counting CPU time
	clock_t finish = clock(); 
	
	double duration = static_cast<double>( (finish - start) / CLOCKS_PER_SEC );

	// Get time as long integer
	time_t long_time;
	time( &long_time );

	// Convert to local time
	struct tm *newtime;
	newtime = localtime( &long_time ); 

	// Set up extension
	char am_pm[] = "AM";
	if( newtime->tm_hour > 12 )        
		strcpy( am_pm, "PM" );

	// Convert from 24-hour to 12-hour clock
	if( newtime->tm_hour > 12 )        
		newtime->tm_hour -= 12;

	// Set hour to 12 if midnight
	if( newtime->tm_hour == 0 )        
		newtime->tm_hour = 12;

	sprintf(message, "\nFine elaborazione\n");
	console.Show(message);
	sprintf(message, "tempo CPU impiegato %5.3f secondi\n", duration);
	console.Show(message);
	sprintf(message, "Feno - Log File %.19s %s\n", asctime( newtime ), am_pm );
	console.Show(message);

	console.Close();
}


