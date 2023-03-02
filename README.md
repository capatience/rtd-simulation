# CPOX-FT RTD Simulation

This python project simulates the residence-time distribution of gases in the CPOX FT reactor setup.
It considers three bodies with volumes V1, V2, and V3.
V1 corresponds to the volume of the upper section of the CPOX reactor, and V2 of lower section of the CPOX reactor.
V3 corresponds to the volume of the FT reactor.

Create venv: python3 -m venv env
Activate the venv: $ source env/bin/activate
Update requirements: $ pip3 freeze > requirements.txt
Install dependencies: $ pip3 install -r requirements.txt
Deactivate venv: $ deactivate

Pipeline is as follows:

1. Update parameters.csv for the simulation you want to run.
2. In main.py, update OUTPUT_FILENAME to what you want.
3. Run main.py.
4. In visualize.py, update INPUT_FILENAME to what OUTPUT_FILENAME was.
5. Run visualize.py.
