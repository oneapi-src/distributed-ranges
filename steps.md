Kroki podjęte:
1. próba
 - wygenerowanie katalogu 'debian' poprzez dh_make
 - modyfikacja plików: control, rules, changelog dla DR(załączone)
 - stworzenie Dockerfile bazującego na oneapi:latest
 - docker build 
 - docker run
 - debuild -us -uc
    - dpkg-source: error: can't build with source format '3.0 (quilt)': no upstream tarball found at ../distributed-ranges_0.1.orig.tar.{bz2,gz,lzma,xz}

    po utworzeniu tez nie pomoglo
    - zasugerowana zmiana https://askubuntu.com/questions/675154/dpkg-buildpackage-dpkg-source-no-upstream-tarball-found - pomogło dodanie '-b -rfakeroot'
- debuild -b -rfakeroot -us -uc
    - problem z MKL
    /distributed-ranges/include/./dr/mhp.hpp:34:10: fatal error: mkl.h: No such file or directory
    34 | #include <mkl.h>
    - próby zmian w /debian/rules np na po prostu dh_auto_configure - ten sam problem z MKL
    - zasugerowane usuniecie source/format - bez zmian

- ponowne ustawianie setvars tez nic nie daje
- MKLROOT ustawiony

2. próba
- budowanie paczki manualnie
- stworzenie katalogu DEBIAN (a nie 'debian'), a w nim manualnie changelog, control oraz rules
- dostosowanie zawartości do narzędzia dpkg-deb
    - dodanie Version, usunięcie Depends, połączenie sekcji co budowania i uruchamiania
- dpkg-deb --build distributed-ranges  (polecenie katalog wyżej)

---------------------------------------
STEPS - źródło https://www.youtube.com/watch?v=ep88vVfzDAo

1. Create the directory to hold the project:

shell) mkdir /home/USER/debpkgs/my-program_version_architecture

2. Create a directory called "DEBIAN" inside the project directory:

shell) mkdir /home/USER/debpkgs/my-program_version_architecture/DEBIAN

3. Copy files into project root directory and include the final paths:

/usr/bin/ would be /home/USER/debpkgs/my-program_version_architecture/usr/bin/

/opt/ would /home/USER/debpkgs/my-program_version_architecture/opt/

4. Create a control file in DEBIAN:

shell) touch /home/USER/debpkgs/my-program_version_architecture/DEBIAN/control

5. Now add the necessary meta data to the control file:

Package: my-program
Version: 1.0
Architecture: all
Essential: no
Priority: optional
Depends: packages;my-program;needs;to;run
Maintainer: Your Name
Description: A short description of my-program that will be displayed when the package is being selected for installation. 

6. If desired a "preinst" and/or "postinst" script can be added that execute before and/or after installation. They must be given proper execute permissions to run:

shell) touch /home/USER/debpkgs/my-program_version_architecture/DEBIAN/postinst

Add commands you'd like to run in postinst and then set the permissions to 755.

7. Now generate package:

shell) dpkg-deb --build /home/USER/debpkgs/my-program_version_architecture



