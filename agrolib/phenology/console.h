#ifndef CONSOLE_H
#define CONSOLE_H

    #include <iostream>

    class Console
    {
    private:
        FILE* m_stream;

    public:
        Console() : m_stream(nullptr) {}

        void Open(const char* filename, const char* mode)
        {
          if( filename != nullptr )
              m_stream = fopen(filename, mode);
        }

        void Close()
        {
          if( m_stream )
              fclose(m_stream);
        }

        void Read(char* message)
        {
          if( m_stream )
              fgets(message, 256, m_stream);
        }

        void Write(char* message)
        {
          if( m_stream )
              fprintf(m_stream, "%s", message);
        }

        int EndOfFile()
        {
          if( m_stream )
              return feof(m_stream);
          return true;
        }

        int Show(char* message)
        {
          if( m_stream )
              return fprintf(m_stream, "%s", message);

          printf("%s", message);
          return false;
        }

        FILE* Stream() { return m_stream; }
    };


    // functions
    void login(const char* log, Console& console);
    void logout(Console& console);

#endif
