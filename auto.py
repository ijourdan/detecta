#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:52:45 2018

@author: ivan
"""

import argparse
from src import detecta

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dispo', default=0, help="DISPO: Camera or video dispositive number. "
                                                                "Integrated cameras commonly use 0 as dispositive number."
                                                                "In case of external camera, try dispositive 1 instead 0 ")
args = vars(ap.parse_args())

a = detecta.Target()
aux = args['dispo']
try:
    aux = int(aux)
    print('Camera Selected: %d' %aux)
except ValueError:
    print('Working on '+aux)

a.start(nomb=aux)
