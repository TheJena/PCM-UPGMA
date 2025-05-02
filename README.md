# PCM-UPGMA
Run UPGMA over some language parameters

## Installation

To use the software in this repository, follow these steps:

1. Clone the repository:

    `git clone https://github.com/TheJena/PCM-UPGMA.git ~/PCM-UPGMA`

1. Create a virtual environment, e.g., with
   [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation):

    ```
    mkvirtualenv -a ~/PCM-UPGMA                 \
                 --creator=venv                 \
                 --download                     \
                 --python=/usr/bin/python3      \
                 -r requirements.txt            \
                 --reset-app-data               \
                 --verbose                      \
                 PCM
    ```

1. Navigate to the project directory and activate the virtual
   environment:

   `workon --cd PCM`

1. Define the password for en/de-cryption:

    `export SECRET="Something you have been told offline/IRL"`

1. Decrypt protected files:

    `make decrypt-all`

1. Run script with -h/--help to discoveer how it should be run/used:

    `python3 ./path_to_a_script/script_name.pt -h`

## LICENSE

See full text license [here](COPYING); what follows are the copyright
and license notices.


```
Copyright (C) 2024-2025 Federico Motta    <federico.motta@unimore.it>
                        Lorenzo  Carletti <lorenzo.carletti@unimore.it>

                   2023 Federico Motta    <federico.motta@unimore.it>
                        Lorenzo  Carletti <lorenzo.carletti@unimore.it>
                        Matteo   Vanzini  <matteo.vanzini@unimore.it>
                        Andrea   Serafini <andrea.serafini@unimore.it>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
