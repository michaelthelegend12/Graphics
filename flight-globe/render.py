#!/usr/bin/env python3
"""Flight globe: London -> Dubai -> Tokyo.

15 s, 1920x1080, 30 fps. A dotted 3D globe rotates east while a plane flies two
great-circle arcs that draw themselves, with city labels popping in.

Needs numpy, scipy, opencv, Pillow and ffmpeg.
    python3 render.py [out.mp4]          full render
    python3 render.py --preview 0.9 6.5  write PNG stills of those times to ./preview
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

W, H, FPS, DUR = 1920, 1080, 30, 15.0
CX, CY, R = 960, 505, 420          # globe centre and radius in pixels
EARTH_KM = 6371.0
LIFT = 1.1                          # screen-space lift for the flight altitude

# ----------------------------------------------------------------------------
# Land outlines (lon, lat). Deliberately coarse: they are only used to decide
# which dots of the globe are land, so small errors disappear in the dot grid.
# ----------------------------------------------------------------------------
EURASIA = [
    (-9.5, 38.7), (-9.3, 43.0), (-8, 43.7), (-1.8, 43.4), (-1.2, 46), (-2.5, 47.2), (-4.7, 48.4),
    (-1.5, 49.7), (1.6, 50.9), (4.5, 52.5), (8.2, 55.5), (8.2, 57.0), (10.5, 57.7), (10.0, 54.5),
    (14, 54.0), (19.5, 54.5), (21, 56.5), (24, 57.5), (24, 59.4), (30, 60.0), (25, 60.3), (22, 60.5),
    (21.5, 63), (25, 65.2), (22, 65.8), (17.5, 62.5), (18.5, 60), (16.5, 57), (13, 55.5), (12.5, 56.5),
    (11, 59), (8, 58), (5.5, 59), (5, 62), (10, 64.5), (14, 67.5), (19, 70), (25, 71), (31, 70),
    (40, 67.5), (41, 66), (44, 68), (53, 68.5), (60, 69), (68, 69), (73, 72.5), (80, 73), (87, 75),
    (100, 77), (105, 77.7), (113, 74), (125, 73), (130, 71), (140, 72.5), (150, 71.5), (160, 69.5),
    (170, 70), (180, 68.5), (180, 65), (172, 61), (164, 60), (162, 57), (160, 53), (156.5, 51),
    (155.8, 54), (156.5, 57), (160, 60.5), (155, 59.3), (143, 59.3), (137, 54), (141, 53), (140, 48),
    (135, 43.5), (131, 42.5), (129.5, 40.5), (129.3, 36), (126.5, 34.5), (126.5, 37.7), (125, 39.5),
    (121.5, 39), (122, 40.5), (118, 39), (118.5, 37.5), (122.5, 37), (119, 35), (121.8, 31.8),
    (122, 30), (120.5, 27), (117, 23.5), (113.5, 22.2), (110, 20.5), (108, 21.5), (106.5, 19),
    (109, 15), (109.3, 11.5), (105, 8.7), (104.8, 10.5), (100.9, 13.5), (100, 12), (99.8, 9),
    (100.5, 6.5), (102.5, 5), (103.5, 2.5), (103.8, 1.3), (101.5, 2.5), (100.3, 5.5), (98.3, 8),
    (98.5, 11), (97.7, 16.5), (94.5, 16), (94, 18.5), (92, 21), (91.5, 22.5), (89, 21.8),
    (86.8, 20.5), (85, 19.5), (80.5, 15.5), (80.2, 13), (79.8, 10.3), (78, 8.8), (77.5, 8),
    (76.3, 9.5), (75, 12.5), (73.5, 16.5), (72.8, 20), (72.7, 21.5), (70.5, 20.8), (69, 22.3),
    (70, 23), (68.2, 23.7), (66.5, 25.3), (62, 25.2), (57.5, 25.7), (56.5, 27.1), (54, 26.7),
    (51, 28.5), (49, 30), (48, 30), (48.2, 29.5), (49.5, 27.3), (50.2, 26.2), (50.9, 25.0),
    (51.6, 24.3), (54, 24.4), (55.1, 25.1), (55.6, 25.5), (56.1, 26.1), (56.4, 26.3), (56.3, 25.4),
    (57.5, 23.9), (58.8, 23.5), (59.8, 22.4), (58.5, 20.5), (57.7, 19.0), (55.5, 17.5), (52.2, 16),
    (49, 14.5), (45, 12.8), (43.3, 12.7), (42.8, 15), (42.7, 17), (40.5, 20.5), (39, 22),
    (37.5, 25), (35, 28), (34.9, 29.5), (34.2, 27.8), (33.5, 28.0), (32.6, 29.9), (32.3, 31.2),
    (34.3, 31.3), (35.8, 34.5), (36, 36.6), (34, 36.2), (30.5, 36.5), (27.5, 36.8), (27, 37.5),
    (26.3, 39.5), (26, 40.7), (24, 40.7), (23.5, 39.5), (24, 38), (23, 36.5), (22.5, 36.5),
    (21.5, 37), (21, 38.5), (20, 39.7), (19.5, 41), (19.5, 42), (18.5, 42.5), (15.5, 45),
    (13.7, 45.2), (12.5, 44.5), (12.5, 44), (14, 42.5), (16, 41.9), (18.5, 40.2), (17, 39.5),
    (16.5, 38.5), (15.7, 38), (16, 39.5), (15.5, 40.2), (14, 41), (12, 42), (10.5, 43), (8.8, 44.4),
    (7, 43.7), (4.5, 43.4), (3, 43), (3.2, 42), (0.5, 40.5), (0, 39.5), (-0.5, 38.5), (-2, 36.7),
    (-5.4, 36.1), (-6.3, 36.9), (-8.9, 37),
]
AFRICA = [
    (32.3, 31.2), (29, 30.9), (25, 31.7), (23, 32.7), (20, 32.2), (19, 30.3), (15.2, 32.3),
    (11, 33), (10, 34), (11, 35), (10.5, 36.8), (8, 37), (3, 36.8), (1, 36.5), (-2, 35.1),
    (-5.9, 35.8), (-6.5, 34.2), (-8.5, 33.3), (-9.7, 31), (-10, 29.5), (-13, 27.7), (-15.5, 24.5),
    (-17, 21), (-16.2, 19), (-16.5, 16.5), (-17.5, 14.7), (-16.7, 12.5), (-15, 11), (-13, 8.5),
    (-11.5, 7), (-9, 5), (-7.5, 4.4), (-4, 5.2), (-1, 4.8), (2, 6.2), (4.5, 6), (6, 4.3),
    (8.5, 4.5), (9.8, 3.2), (9.5, 1), (9, -1), (12, -5), (12.2, -6), (13.2, -9), (13.5, -12),
    (11.8, -16), (12.3, -18), (14.5, -22.5), (15, -26.5), (16.5, -28.5), (18, -32), (18.4, -34.2),
    (20, -34.8), (22, -34.2), (25.5, -34), (27, -33.5), (30.5, -31), (32.9, -28), (32.8, -25.8),
    (35, -24), (35, -20), (37, -18), (40.5, -15), (39.5, -10), (39.2, -6), (40, -3), (42, -1),
    (46, 2), (49, 6), (51, 10), (51, 11.8), (45, 10.5), (43.3, 11.8), (42.5, 13), (40, 15.5),
    (38.5, 18), (37.2, 21), (35.5, 24), (33.8, 27.3), (32.6, 29.9),
]
NORTH_AMERICA = [
    (-168, 65.7), (-163, 69), (-156.5, 71.3), (-141, 69.7), (-128, 70), (-115, 68), (-108, 68),
    (-96, 68.5), (-94, 71.5), (-90, 69), (-85, 69.5), (-82, 67), (-86, 66), (-90, 64), (-93, 61.5),
    (-94.5, 59), (-92.5, 57), (-88, 56.3), (-82, 55.5), (-82, 52.8), (-79.5, 51.5), (-78.5, 54),
    (-76.5, 56.5), (-77.5, 59), (-78, 62), (-72, 61.5), (-70, 61), (-68, 58.5), (-65, 60.5),
    (-62, 57), (-59.5, 54), (-56, 52), (-55.7, 51.5), (-60, 50.2), (-64.5, 49.2), (-66.5, 50),
    (-68.5, 48.8), (-70.5, 47), (-65, 48.8), (-64.5, 46.2), (-66.2, 45), (-67, 44.7), (-70.5, 43),
    (-70.8, 42.2), (-70, 41.7), (-72, 41), (-74, 40.5), (-74, 39), (-75.5, 38), (-76, 36.9),
    (-75.5, 35.3), (-77.5, 34.2), (-79.5, 33), (-81, 31.5), (-81.4, 30), (-80.5, 28.5), (-80, 26.8),
    (-80.3, 25.2), (-81.2, 25.2), (-82.8, 27.5), (-82.7, 29.2), (-84, 30), (-85.5, 29.7),
    (-87.5, 30.3), (-89.2, 30.2), (-89.4, 29), (-91, 29.2), (-93.5, 29.8), (-95, 29), (-97.2, 27.7),
    (-97.5, 25.5), (-97.7, 22), (-96, 19), (-94.5, 18.2), (-92, 18.6), (-90.5, 20.8), (-88, 21.5),
    (-87, 21.4), (-87.5, 18.5), (-88.5, 15.8), (-86, 15.8), (-83.5, 15.2), (-83.3, 12.5),
    (-83.7, 10.9), (-82.5, 9.2), (-81, 8.9), (-79.5, 9.5), (-77.5, 8.5), (-77.9, 7.2), (-80, 7.5),
    (-81, 8), (-83.2, 8.4), (-85.7, 10), (-87, 12.5), (-88, 13.2), (-91, 14), (-94, 16),
    (-96.5, 15.7), (-100, 17), (-105, 19.5), (-105.5, 22.5), (-108.5, 25.5), (-110.8, 28.5),
    (-112.8, 31.2), (-114.8, 31.8), (-114.7, 30), (-113, 28), (-111.5, 26), (-110, 24),
    (-109.5, 23), (-110.5, 23.2), (-112, 24.8), (-114.2, 27.2), (-115.7, 29.5), (-116.8, 32.2),
    (-117.2, 32.7), (-118.5, 34), (-120.5, 34.6), (-121.9, 36.6), (-122.5, 37.7), (-123.8, 39.5),
    (-124.4, 40.4), (-124, 43), (-124, 46.2), (-124.7, 48.3), (-123.5, 49), (-125, 49.5),
    (-128, 51), (-130.5, 54.5), (-133, 57), (-136, 58.2), (-140, 59.8), (-144, 60), (-148, 60.5),
    (-151, 59.2), (-153, 58), (-157, 56.5), (-162, 55), (-164.5, 54.5), (-161, 58.5), (-160, 56),
    (-158, 58.5), (-162, 58.7), (-165, 60.5), (-164.5, 63), (-161, 64.5), (-166, 64.6),
]
SOUTH_AMERICA = [
    (-77.3, 8.7), (-76, 9.5), (-75.5, 10.7), (-73, 11.3), (-71.5, 12.4), (-71, 11), (-69.5, 11.5),
    (-67, 10.6), (-64, 10.5), (-62, 10.7), (-60.5, 8.5), (-57, 6), (-54, 5.8), (-51.5, 4.2),
    (-50, 1.5), (-50, 0), (-48, -0.8), (-44.5, -2.5), (-41, -2.9), (-38.5, -3.7), (-35.2, -5.6),
    (-34.8, -8), (-37, -11.5), (-38.8, -13.5), (-39, -17.5), (-40.5, -20.5), (-41.8, -23),
    (-44.5, -23.2), (-48, -26), (-48.7, -28.5), (-50.5, -31), (-53.5, -34), (-57, -34.5),
    (-58.5, -34.2), (-57.5, -38), (-62, -39), (-62.3, -41), (-65, -41), (-64.5, -42.5), (-65.5, -45),
    (-67.5, -46.5), (-66, -48), (-69, -51), (-68.5, -52.4), (-68, -55), (-71.5, -54), (-74.5, -51.5),
    (-75.5, -47.5), (-73.5, -43.5), (-73.7, -39.5), (-72.5, -35), (-71.5, -30), (-70.5, -23),
    (-70.2, -18.5), (-76, -14), (-79, -8), (-81.2, -6), (-80.3, -3.5), (-80, -1), (-80, 1),
    (-78.8, 1.8), (-77.5, 4), (-77.4, 6.5), (-77.9, 7.2),
]
AUSTRALIA = [
    (113.5, -22), (114, -26), (115, -33.5), (118, -35), (123.5, -34), (126, -32.2), (131, -31.5),
    (134, -32.7), (135.8, -34.9), (137.7, -32.5), (138.2, -35), (140, -37.5), (143.5, -38.8),
    (146.3, -39), (148, -37.8), (150, -37), (151.3, -33.8), (153.2, -30), (153.5, -27),
    (152.5, -24.5), (150.5, -22.5), (149, -20.5), (146.3, -19), (145.5, -15), (143.5, -14),
    (142.5, -10.7), (141.5, -13), (141.5, -17), (139.5, -17.5), (135.5, -15), (136.8, -12.2),
    (132.5, -11.5), (130, -12.4), (129.5, -15), (127, -14), (125, -15), (122, -17.5), (121, -19.5),
    (118, -20.5), (116, -20.8), (114, -22),
]
GREENLAND = [
    (-73, 78.5), (-60, 82), (-40, 83.5), (-22, 82.5), (-18, 80), (-20, 76), (-22, 72), (-26, 69),
    (-34, 66.5), (-40, 65), (-42, 62), (-44, 60), (-48, 61), (-51, 64), (-54, 67), (-54, 70),
    (-56, 73), (-60, 76), (-68, 76.2),
]
ANTARCTICA = [
    (-180, -74), (-160, -77.5), (-150, -75.5), (-135, -74.5), (-120, -74), (-100, -73.5),
    (-80, -73.5), (-75, -72), (-68, -68), (-62, -64.5), (-57, -63.5), (-60, -66), (-66, -70),
    (-62, -73), (-58, -75), (-45, -78), (-30, -78), (-10, -71), (10, -70), (30, -69.5),
    (50, -67.5), (70, -68), (75, -69.5), (90, -66.5), (110, -66), (130, -66), (145, -68),
    (160, -70.5), (170, -72), (180, -74), (180, -90), (-180, -90),
]
ISLANDS = [
    [(-5.7, 50.1), (-3, 50.6), (1.3, 51.1), (1.7, 52.7), (0, 53.6), (-0.5, 54.5), (-1.5, 55.5),
     (-2, 56), (-2, 57.5), (-3.5, 57.7), (-4, 58.6), (-5, 58.6), (-6, 57.5), (-5.6, 56), (-5, 55),
     (-4.8, 54.8), (-3, 54.8), (-3, 53.5), (-4.7, 53.3), (-4.2, 52.4), (-5.2, 51.8), (-3, 51.4),
     (-4.5, 51.0)],                                                                  # Great Britain
    [(-10, 51.7), (-6, 52), (-6, 54), (-5.5, 54.7), (-7.5, 55.3), (-10, 54), (-9.5, 52.3)],  # Ireland
    [(-24, 65.5), (-22, 66.4), (-16, 66.5), (-13.5, 65), (-18, 63.4), (-22.5, 63.8)],        # Iceland
    [(130.9, 34), (132.5, 35.4), (133.5, 35.5), (135.5, 35.7), (136.8, 37.3), (138.5, 37.3),
     (139.8, 39.5), (140, 41), (141.5, 41.3), (142, 39.5), (141, 38.3), (140.7, 36.9),
     (140.9, 35.7), (140, 34.95), (139.3, 35.2), (138.8, 34.6), (137.5, 34.6), (136.9, 34.7),
     (136.3, 34.2), (135.1, 33.5), (135.2, 34.6), (134.5, 34.1), (132.5, 34.2), (131, 33.9)],  # Honshu
    [(129.7, 33), (130.5, 33.9), (131.8, 33.3), (131.5, 31.5), (130.7, 31), (130.2, 32)],       # Kyushu
    [(132.5, 33.5), (134.3, 34.2), (134.5, 33.5), (133, 32.8)],                                  # Shikoku
    [(140, 41.5), (141.5, 43.2), (141.5, 45.4), (143.5, 44), (145.5, 43.3), (143, 42),
     (141.5, 42.5), (140.4, 42.5)],                                                           # Hokkaido
    [(142, 46), (143.2, 49), (143, 53.5), (142, 54), (142.2, 50)],                            # Sakhalin
    [(120.2, 22.9), (121.5, 25.2), (122, 24.8), (120.9, 22)],                                 # Taiwan
    [(108.7, 19.2), (110.5, 20), (111, 19.5), (109.5, 18.3)],                                 # Hainan
    [(79.8, 9.5), (81.8, 7.5), (81, 6), (80, 6.2), (79.8, 8)],                                # Sri Lanka
    [(95.3, 5.6), (98.5, 4), (104, -1), (106, -3), (106, -5.8), (104.5, -5.9), (101.5, -2.8),
     (98.5, 1.7), (95.5, 3.8)],                                                               # Sumatra
    [(109, 1.5), (111, 2), (113, 3.2), (115, 5), (117, 7), (119, 5), (118, 2), (117.5, 0.7),
     (116, -3.5), (114.5, -4), (111, -3), (110, -1.5), (109, -0.5)],                          # Borneo
    [(105.2, -6.8), (108, -6.3), (111, -6.5), (114.5, -7.8), (114, -8.7), (108, -7.8),
     (106, -7.4)],                                                                            # Java
    [(119, 0.5), (121, 1.3), (124.5, 1.7), (125, 1), (122.5, 0), (123, -1), (121.5, -1.5),
     (122.5, -4.5), (121, -4.7), (120.5, -5.5), (119.5, -5.5), (119.4, -3)],                  # Sulawesi
    [(131, -0.8), (135, -3.2), (138, -1.7), (141, -2.6), (145, -4.5), (148, -6), (147.5, -8),
     (150, -10.3), (147, -10), (144, -7.8), (141, -9.2), (138, -8.3), (137.8, -5.5), (134, -4),
     (132, -2.7)],                                                                            # New Guinea
    [(120, 18.5), (122.2, 18.5), (122, 16), (124, 13.5), (121.5, 13), (120, 14.5)],           # Luzon
    [(122, 7), (125.5, 9.7), (126.5, 7), (125.5, 5.8), (123.5, 7.5)],                         # Mindanao
    [(144.7, -40.7), (148.3, -40.9), (148, -43.2), (146.2, -43.6)],                           # Tasmania
    [(172.7, -34.4), (174.5, -36.5), (175.5, -37), (178.5, -37.6), (177, -39.2), (175.3, -41.6),
     (174.7, -41), (173.8, -39.3), (174.7, -38), (173, -35.5)],                               # NZ north
    [(172.7, -40.5), (174, -41.3), (173, -43.5), (171, -44.5), (169, -46.5), (166.5, -46),
     (168, -44), (170.5, -42.8), (171.5, -41.7)],                                             # NZ south
    [(49.3, -12), (50.4, -15.5), (47.5, -24.5), (45, -25.5), (43.7, -22), (44.3, -17),
     (46.5, -15.5)],                                                                          # Madagascar
    [(-84.9, 21.9), (-82.5, 23.1), (-80, 23), (-77.5, 21.7), (-74.2, 20.2), (-77.5, 19.9),
     (-79.5, 21.5), (-82, 22.2)],                                                             # Cuba
    [(-74.4, 18.4), (-72.5, 19.9), (-69.5, 19.3), (-68.4, 18.6), (-71, 17.8)],                # Hispaniola
    [(-90, 73), (-80, 73.8), (-72, 71.5), (-67, 69), (-62, 66.5), (-64, 63.5), (-68, 62.5),
     (-72, 63.8), (-77, 64.5), (-74, 67.5), (-81, 69.5), (-90, 70.5)],                        # Baffin
    [(-118, 72), (-105, 73), (-100, 70.5), (-108, 68.5), (-117, 69.5)],                       # Victoria
    [(-90, 76.5), (-80, 76.3), (-76, 79), (-68, 82), (-62, 82.5), (-75, 83), (-90, 81.5),
     (-95, 79)],                                                                              # Ellesmere
    [(51, 71), (55, 73.5), (68, 76.8), (60, 75), (56, 71.5)],                                 # Novaya Zemlya
    [(11, 79), (17, 80.2), (27, 80), (22, 77.5), (15, 77)],                                   # Svalbard
    [(-180, 68.5), (-175, 67.5), (-170, 66), (-170, 65.5), (-176, 65), (-180, 65)],           # Chukotka
]
BLACK_SEA = [(28, 41.5), (27.8, 43.5), (29.5, 45.3), (31, 46.5), (33.5, 46), (32.5, 45.3), (34, 44.6),
             (36.5, 45.4), (38, 47), (39.5, 46.8), (38, 45), (40, 43.5), (41.6, 42), (41.5, 41.2),
             (36, 41.7), (33, 42), (31, 41.1), (29, 41.1)]
CASPIAN = [(47, 45.2), (49, 46.5), (51, 47), (53, 46.5), (53.5, 45), (51, 44.5), (50.5, 42),
           (52.5, 41.5), (53, 40), (54, 37.5), (52, 36.7), (49.5, 37.5), (49, 39.5), (48.5, 41.5),
           (47.5, 43)]

LAND = [EURASIA, AFRICA, NORTH_AMERICA, SOUTH_AMERICA, AUSTRALIA, GREENLAND, ANTARCTICA] + ISLANDS
WATER = [BLACK_SEA, CASPIAN]

# ----------------------------------------------------------------------------
# Route
# ----------------------------------------------------------------------------
CITIES = [
    ("LONDON", "United Kingdom", "LON", 51.5074, -0.1278, (0, -88), 0.80),
    ("DUBAI", "United Arab Emirates", "DXB", 25.2048, 55.2708, (0, 92), 6.00),
    ("TOKYO", "Japan", "TYO", 35.6762, 139.6503, (70, 88), 11.20),
]
T_LEG = [(1.4, 6.4), (6.6, 11.6)]          # take-off and landing time of each leg
T_PLANE_IN, T_PLANE_OUT = 1.05, 11.6       # plane appears / lands
T_TITLE = 12.2

ACCENT = [(0.0, (255, 214, 102)), (0.5, (255, 140, 66)), (1.0, (255, 82, 128))]


def route_color(u):
    u = min(max(u, 0.0), 1.0)
    for (a, ca), (b, cb) in zip(ACCENT, ACCENT[1:]):
        if u <= b:
            k = (u - a) / (b - a)
            return tuple(ca[i] + (cb[i] - ca[i]) * k for i in range(3))
    return ACCENT[-1][1]


def lonlat_xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


CITY_XYZ = [lonlat_xyz(c[4], c[3]) for c in CITIES]
LEGS = []
for _i in range(2):
    a, b = CITY_XYZ[_i], CITY_XYZ[_i + 1]
    om = math.acos(float(np.clip(a @ b, -1, 1)))
    LEGS.append((a, b, om, 0.14 * om))
LEG_KM = [l[2] * EARTH_KM for l in LEGS]
TOTAL_KM = sum(LEG_KM)
FRAC = [0.0, LEG_KM[0] / TOTAL_KM, 1.0]


def leg_points(i, s):
    a, b, om, h = LEGS[i]
    s = np.atleast_1d(np.asarray(s, float))
    p = (np.sin((1 - s) * om)[:, None] * a + np.sin(s * om)[:, None] * b) / math.sin(om)
    return p * (1 + h * np.sin(math.pi * s))[:, None]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def leg_s(i, t):
    a, b = T_LEG[i]
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    return 0.45 * u + 0.55 * smooth(u)


def ease_out_back(x):
    if x <= 0.0:
        return 0.0
    x = min(x, 1.0)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def active_leg(t):
    return 0 if t < 6.5 else 1


def plane_xyz(t):
    i = active_leg(t)
    return leg_points(i, leg_s(i, t))[0], i


# ----------------------------------------------------------------------------
# Camera: the globe turns east the whole time, tilting a little with the plane
# ----------------------------------------------------------------------------
_lon_keys = PchipInterpolator(
    [0, 1.4, 6.4, 6.6, 11.6, 15.0], [-32, -14, 48, 50, 128, 141])
_tg = np.arange(0, DUR + 0.001, 0.05)
_plat = []
for _t in _tg:
    _p, _ = plane_xyz(_t)
    _plat.append(math.degrees(math.asin(_p[2] / np.linalg.norm(_p))))
_lat_raw = 0.55 * np.array(_plat) + 12.0
_lat_smooth = gaussian_filter1d(_lat_raw, 14, mode="nearest")
_lat_cam = PchipInterpolator(_tg, np.clip(_lat_smooth, 24, 38))


def camera(t):
    return float(_lon_keys(t)), float(_lat_cam(t))


def view_xyz(P, lon0, lat0):
    l0, p0 = math.radians(lon0), math.radians(lat0)
    c, s = math.cos(l0), math.sin(l0)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    xv = -s * X + c * Y
    u = c * X + s * Y
    yv = math.cos(p0) * Z - math.sin(p0) * u
    zv = math.sin(p0) * Z + math.cos(p0) * u
    return xv, yv, zv


def project(P, cam):
    xv, yv, zv = view_xyz(P, *cam)
    vis = (zv > 0) | (xv * xv + yv * yv > 1.0)
    lift = (np.linalg.norm(P, axis=-1) - 1.0) * R * LIFT * (0.25 + 0.75 * np.clip(zv, 0, 1))  # raise high points up the screen
    return CX + R * xv, CY - R * yv - lift, zv, vis


# ----------------------------------------------------------------------------
# Static assets
# ----------------------------------------------------------------------------
def raster(polys):
    m = np.zeros((720, 1440), np.uint8)
    for p in polys:
        pts = np.array([[(lo + 180) * 4, (90 - la) * 4] for lo, la in p], np.float32)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    return m


def build_land_dots(n=24000):
    mask = raster(LAND)
    mask[raster(WATER) > 0] = 0
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * i / n))
    lon = (np.degrees(i * math.pi * (3 - math.sqrt(5))) + 180) % 360 - 180
    px = np.clip(((lon + 180) * 4).astype(int), 0, 1439)
    py = np.clip(((90 - lat) * 4).astype(int), 0, 719)
    keep = mask[py, px] > 0
    return lonlat_xyz(lon[keep], lat[keep])


LAND_DOTS = build_land_dots()
LIGHT = np.array([-0.38, 0.46, 0.80])
LIGHT /= np.linalg.norm(LIGHT)


def build_base():
    rng = np.random.RandomState(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2) / R
    k = np.clip(np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H / 2) / (H * 0.78)) ** 2), 0, 1)
    c_in, c_out = np.array([15, 24, 56], np.float32), np.array([3, 5, 14], np.float32)
    bg = c_in[None, None, :] + (c_out - c_in)[None, None, :] * (k ** 0.9)[..., None]
    for _ in range(300):                                    # stars
        x, y = rng.randint(0, W), rng.randint(0, H)
        v = rng.randint(70, 220)
        bg[y, x] = np.maximum(bg[y, x], v)
        if rng.rand() < 0.2:
            bg[min(y + 1, H - 1), x] = np.maximum(bg[min(y + 1, H - 1), x], v * 0.5)
            bg[y, min(x + 1, W - 1)] = np.maximum(bg[y, min(x + 1, W - 1)], v * 0.5)
    glow = np.exp(-np.clip(d - 1, 0, None) * R / 42.0)       # atmosphere outside the disc
    bg += (glow * (d > 1))[..., None] * np.array([40, 115, 255], np.float32) * 0.55
    nx, ny = (xx - CX) / R, -(yy - CY) / R
    nz = np.sqrt(np.clip(1 - d * d, 0, 1))
    diff = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    ocean = np.array([8, 20, 48], np.float32) + np.array([20, 58, 118], np.float32) * (diff ** 1.1)[..., None]
    rim = ((1 - nz) ** 3)[..., None] * np.array([35, 100, 215], np.float32) * 0.65
    hv = LIGHT + np.array([0, 0, 1.0])
    hv /= np.linalg.norm(hv)
    spec = (np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 45)[..., None] * np.array([80, 110, 160], np.float32) * 0.35
    globe = ocean + rim + spec
    alpha = np.clip((1 - d) * R + 0.5, 0, 1)[..., None]
    img = bg * (1 - alpha) + globe * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


BASE = build_base()

POPPINS_B = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
POPPINS_M = "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"
for _f in (POPPINS_B, POPPINS_M):
    if not os.path.exists(_f):
        POPPINS_B = POPPINS_M = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def make_label(name, sub):
    """City label at 2x size. Returns premultiplied RGB (float) and alpha (float)."""
    f1, f2 = font(POPPINS_B, 78), font(POPPINS_M, 34)
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    w1 = tmp.textlength(name, font=f1)
    w2 = tmp.textlength(sub, font=f2)
    pad_x, pad_y = 56, 30
    w = int(max(w1, w2) + 2 * pad_x)
    h = int(78 + 34 + 2 * pad_y + 20)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle((2, 2, w - 3, h - 3), radius=40, fill=(8, 14, 34, 228), outline=(255, 190, 100, 255), width=4)
    dr.text((w / 2, pad_y + 52), name, font=f1, fill=(255, 255, 255, 255), anchor="ms")
    dr.text((w / 2, pad_y + 52 + 14 + 34), sub, font=f2, fill=(165, 190, 230, 255), anchor="ms")
    a = np.asarray(im).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] * 255.0
    return rgb, a[..., 3]


LABELS = [make_label(c[0], c[1]) for c in CITIES]


def make_pill(parts, tail, size=34):
    """Title pill: city names joined by drawn arrows (Poppins has no arrow glyph), then a tail text."""
    f = font(POPPINS_B, size * 2)
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    gap, arrow = 44, 56
    widths = [tmp.textlength(p, font=f) for p in parts]
    tail_w = tmp.textlength(tail, font=f)
    total = sum(widths) + (len(parts) - 1) * (2 * gap + arrow) + 2 * gap + 14 + tail_w
    w, h = int(total + 110), int(size * 2 + 56)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle((2, 2, w - 3, h - 3), radius=h // 2, fill=(8, 14, 34, 232), outline=(255, 190, 100, 255), width=4)
    x, cy = 55.0, h / 2
    for i, p in enumerate(parts):
        dr.text((x, cy + 2), p, font=f, fill=(255, 255, 255, 255), anchor="lm")
        x += widths[i]
        if i < len(parts) - 1:
            x += gap
            col = tuple(int(v) for v in route_color((i + 0.5) / (len(parts) - 1))) + (255,)
            dr.line((x, cy, x + arrow, cy), fill=col, width=6)
            dr.polygon([(x + arrow + 4, cy), (x + arrow - 14, cy - 14), (x + arrow - 14, cy + 14)], fill=col)
            x += arrow + gap
    x += gap - 10
    dr.ellipse((x - 6, cy - 6, x + 6, cy + 6), fill=(255, 190, 100, 255))
    dr.text((x + 30, cy + 2), tail, font=f, fill=(255, 255, 255, 255), anchor="lm")
    a = np.asarray(im).astype(np.float32) / 255.0
    return a[..., :3] * a[..., 3:4] * 255.0, a[..., 3]


TITLE_PILL = make_pill(["LONDON", "DUBAI", "TOKYO"], f"{TOTAL_KM:,.0f} KM".replace(",", " "))


def code_sprite(text, rgb):
    f = font(POPPINS_B, 44)
    tmp = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    w, h = int(tmp.textlength(text, font=f)) + 8, 64
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((w / 2, h / 2), text, font=f, fill=rgb + (255,), anchor="mm")
    a = np.asarray(im).astype(np.float32) / 255.0
    return a[..., :3] * a[..., 3:4] * 255.0, a[..., 3]


CODES_DIM = [code_sprite(c[2], (110, 130, 170)) for c in CITIES]
CODES_ON = [code_sprite(c[2], (255, 255, 255)) for c in CITIES]
FONT_NUM = font(POPPINS_B, 30)
FONT_TINY = font(POPPINS_M, 18)

PLANE = np.array([
    (1.0, 0), (0.80, 0.07), (0.30, 0.09), (-0.05, 0.62), (-0.20, 0.62), (-0.12, 0.10), (-0.62, 0.08),
    (-0.78, 0.30), (-0.90, 0.30), (-0.88, 0.07), (-1.0, 0.04), (-1.0, -0.04), (-0.88, -0.07),
    (-0.90, -0.30), (-0.78, -0.30), (-0.62, -0.08), (-0.12, -0.10), (-0.20, -0.62), (-0.05, -0.62),
    (0.30, -0.09), (0.80, -0.07)], np.float32)


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
def blit(img, sprite, x, y, scale, gain):
    """Alpha-blend a premultiplied sprite (drawn at 2x) with its top-left at (x, y)."""
    rgb, a = sprite
    sw, sh = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
    if gain <= 0.003 or scale <= 0.01:
        return
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
    x0, y0 = int(round(x)), int(round(y))
    ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    r = rgb[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] * gain
    al = (a[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] * gain)[..., None]
    roi = img[iy0:iy1, ix0:ix1].astype(np.float32)
    img[iy0:iy1, ix0:ix1] = np.clip(r + roi * (1 - al), 0, 255).astype(np.uint8)


def visible_runs(sx, sy, vis):
    runs, cur = [], []
    for x, y, v in zip(sx, sy, vis):
        if v:
            cur.append((x, y))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def poly(img, pts, color, th=1):
    if len(pts) >= 2:
        arr = np.round(np.array(pts, np.float32) * 16).astype(np.int32)
        cv2.polylines(img, [arr], False, color, th, cv2.LINE_AA, shift=4)


def line(img, p, q, color, th):
    cv2.line(img, (int(round(p[0] * 16)), int(round(p[1] * 16))),
             (int(round(q[0] * 16)), int(round(q[1] * 16))), color, th, cv2.LINE_AA, shift=4)


def circle(img, c, r, color, th=-1):
    cv2.circle(img, (int(round(c[0] * 16)), int(round(c[1] * 16))), max(1, int(round(r * 16))),
               color, th, cv2.LINE_AA, shift=4)


def sphere_ring(center, rho, n=72):
    c = center / np.linalg.norm(center)
    up = np.array([0, 0, 1.0])
    e1 = np.cross(up, c)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    th = np.linspace(0, 2 * math.pi, n)[:, None]
    return (math.cos(rho) * c + math.sin(rho) * (np.cos(th) * e1 + np.sin(th) * e2)) * 1.003


# ----------------------------------------------------------------------------
# Frame
# ----------------------------------------------------------------------------
def render_frame(t):
    cam = camera(t)
    img = BASE.copy()
    fx = np.zeros_like(img)                                  # additive full-res layer
    glow = np.zeros((H // 2, W // 2, 3), np.uint8)           # additive blurred layer

    # graticule
    for la in range(-60, 61, 30):
        P = lonlat_xyz(np.linspace(-180, 180, 145), np.full(145, la))
        sx, sy, zv, vis = project(P, cam)
        for run in visible_runs(sx, sy, zv > 0):
            poly(img, run, (30, 62, 112), 1)
    for lo in range(-180, 180, 30):
        P = lonlat_xyz(np.full(91, lo), np.linspace(-90, 90, 91))
        sx, sy, zv, vis = project(P, cam)
        for run in visible_runs(sx, sy, zv > 0):
            poly(img, run, (30, 62, 112), 1)

    # land dots
    xv, yv, zv = view_xyz(LAND_DOTS, *cam)
    idx = np.nonzero(zv > 0.03)[0]
    diff = np.clip(xv[idx] * LIGHT[0] + yv[idx] * LIGHT[1] + zv[idx] * LIGHT[2], 0, 1)
    br = 0.22 + 0.78 * diff ** 0.8
    br *= np.clip(zv[idx] * 1.6, 0, 1) ** 0.5
    dark = np.array([28, 70, 128], np.float32)
    lite = np.array([150, 225, 255], np.float32)
    cols = dark[None] + (lite - dark)[None] * br[:, None]
    rad = 3.0 * (0.45 + 0.55 * zv[idx])
    sxs, sys_ = CX + R * xv[idx], CY - R * yv[idx]
    for k in range(len(idx)):
        c = cols[k]
        cv2.circle(img, (int(sxs[k] * 16), int(sys_[k] * 16)), int(rad[k] * 16),
                   (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA, shift=4)

    # arcs
    for i in range(2):
        if i == 0:
            s_head = leg_s(0, t) if t >= T_LEG[0][0] else 0.0
        else:
            s_head = leg_s(1, t) if t >= T_LEG[1][0] else 0.0
        if s_head <= 0.002:
            continue
        n = int(max(3, 300 * s_head))
        ss = np.linspace(0, s_head, n)
        P = leg_points(i, ss)
        sx, sy, zv_, vis = project(P, cam)
        for j in range(n - 1):
            if not (vis[j] and vis[j + 1]):
                continue
            u = FRAC[i] + ss[j] * (FRAC[i + 1] - FRAC[i])
            col = np.array(route_color(u), np.float32)
            head = max(0.0, 1 - (s_head - ss[j]) / 0.10) ** 1.5 * 0.75
            core = col + (255 - col) * head
            line(img, (sx[j], sy[j]), (sx[j + 1], sy[j + 1]), tuple(int(v) for v in core), 4)
            line(glow, (sx[j] / 2, sy[j] / 2), (sx[j + 1] / 2, sy[j + 1] / 2),
                 tuple(int(v * 0.8) for v in col), 9)

    # pins, pulse rings and label data
    label_jobs = []
    for ci, (name, sub, code, lat, lon, off, tpop) in enumerate(CITIES):
        pp = ease_out_back((t - tpop) / 0.5)
        if pp < 0.02:
            continue
        C = CITY_XYZ[ci] * 1.004
        sx, sy, zv_, vis = project(C[None], cam)
        face = float(np.clip((zv_[0] - 0.20) / 0.25, 0, 1))
        if face <= 0:
            continue
        pos = (float(sx[0]), float(sy[0]))
        for ph in (0.0, 0.5):                                # two pulse rings, 1.8 s cycle
            k = (((t - tpop) / 1.8) + ph) % 1.0
            ring = sphere_ring(C, 0.012 + 0.085 * k)
            rx, ry, rz, rv = project(ring, cam)
            amp = (1 - k) ** 1.6 * min(1.0, (t - tpop) / 0.4) * face
            colr = tuple(int(v * amp) for v in (255, 200, 120))
            for run in visible_runs(rx, ry, rz > 0):
                poly(fx, run, colr, 2)
        circle(glow, (pos[0] / 2, pos[1] / 2), 11 * pp, tuple(int(v * face) for v in (200, 150, 70)))
        circle(img, pos, 9.5 * pp, (255, 190, 100), 2)
        circle(img, pos, 4.8 * pp, (255, 255, 255), -1)
        label_jobs.append((ci, pos, face, pp, off))

    # plane
    if T_PLANE_IN <= t < 12.0:
        P, i = plane_xyz(t)
        s = leg_s(i, t)
        grow = smooth((t - T_PLANE_IN) / 0.35) * (1 - smooth((t - T_PLANE_OUT) / 0.4))
        sx, sy, zv_, vis = project(P[None], cam)
        pos = np.array([sx[0], sy[0]])
        eps = 0.004
        a = leg_points(i, min(1.0, s + eps))[0]
        b = leg_points(i, max(0.0, s - eps))[0]
        ax, ay, _, _ = project(np.stack([a, b]), cam)
        ang = math.atan2(ay[0] - ay[1], ax[0] - ax[1])
        alt = math.sin(math.pi * s)
        # shadow on the surface under the plane
        G = P / np.linalg.norm(P)
        gx, gy, gz, _ = project(G[None], cam)
        if gz[0] > 0.05:
            r0 = int(30 * grow) + 2
            x0, y0 = int(gx[0]) - r0 - 4, int(gy[0]) - r0 - 4
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x0 + 2 * r0 + 12), min(H, y0 + 2 * r0 + 12)
            roi = img[y0:y1, x0:x1]
            ov = roi.copy()
            cv2.ellipse(ov, (int(gx[0]) - x0, int(gy[0]) - y0), (int(26 * grow) + 1, int(9 * grow) + 1),
                        math.degrees(ang), 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.38 * min(1.0, alt * 3 + 0.0), roi, 1 - 0.38 * min(1.0, alt * 3 + 0.0), 0, roi)
        circle(glow, (pos[0] / 2, pos[1] / 2), 13 * grow, (95, 115, 160))
        sc = 30 * (1 + 0.35 * alt) * grow
        ca, sa = math.cos(ang), math.sin(ang)
        rot = np.array([[ca, sa], [-sa, ca]], np.float32)
        pts = PLANE * sc @ rot + pos.astype(np.float32)
        arr = np.round(pts * 16).astype(np.int32)
        cv2.fillPoly(img, [arr], (255, 255, 255), cv2.LINE_AA, shift=4)
        cv2.polylines(img, [arr], True, (150, 170, 210), 1, cv2.LINE_AA, shift=4)

    # additive layers
    g = cv2.GaussianBlur(glow, (0, 0), 6)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    img = cv2.add(img, g)
    img = cv2.add(img, fx)

    # labels last, so they sit above the arcs
    for ci, pos, face, pp, off in label_jobs:
        rgb, al = LABELS[ci]
        sc = max(0.0, pp) * 0.5
        w, h = rgb.shape[1] * sc, rgb.shape[0] * sc
        ax, ay = pos[0] + off[0], pos[1] + off[1]
        gain = face * min(1.0, pp * 1.6)
        if off[1] < 0:
            line(img, (pos[0], pos[1] - 12 * pp), (ax, ay), tuple(int(v * gain) for v in (255, 190, 100)), 2)
            blit(img, LABELS[ci], ax - w / 2, ay - h, sc, gain)
        else:
            line(img, (pos[0], pos[1] + 12 * pp), (ax, ay), tuple(int(v * gain) for v in (255, 190, 100)), 2)
            blit(img, LABELS[ci], ax - w / 2, ay, sc, gain)

    draw_hud(img, t)

    if t < 0.8:
        img = cv2.convertScaleAbs(img, alpha=smooth(t / 0.8))
    return img


def flown_km(t):
    km = 0.0
    for i in range(2):
        km += LEG_KM[i] * (leg_s(i, t) if t >= T_LEG[i][0] else 0.0)
    return km


def draw_hud(img, t):
    fade = smooth((t - 0.7) / 0.5)
    if fade > 0:
        x0, x1, y = 610, 1310, 1010
        nodes = [x0 + (x1 - x0) * f for f in FRAC]
        line(img, (x0, y), (x1, y), (52, 66, 100), 4)
        frac = flown_km(t) / TOTAL_KM
        steps = 60
        for k in range(steps):
            u0, u1 = k / steps, (k + 1) / steps
            if u0 >= frac:
                break
            col = tuple(int(v) for v in route_color(u0))
            line(img, (x0 + (x1 - x0) * u0, y), (x0 + (x1 - x0) * min(u1, frac), y), col, 4)
        reached = [t >= 0.8 + 0.0, t >= T_LEG[0][1] - 0.05, t >= T_LEG[1][1] - 0.05]
        for i, nx in enumerate(nodes):
            if reached[i]:
                circle(img, (nx, y), 11, tuple(int(v) for v in route_color(FRAC[i])), -1)
                circle(img, (nx, y), 4.5, (255, 255, 255), -1)
                spr = CODES_ON[i]
            else:
                circle(img, (nx, y), 11, (14, 22, 44), -1)
                circle(img, (nx, y), 11, (90, 108, 150), 2)
                spr = CODES_DIM[i]
            blit(img, spr, nx - spr[0].shape[1] * 0.25, y + 20, 0.5, fade)
        # distance read-out above the bar
        pil = Image.fromarray(img[944:984, 760:1160])
        dr = ImageDraw.Draw(pil)
        dr.text((200, 20), f"{flown_km(t):,.0f} KM FLOWN".replace(",", " "), font=FONT_NUM,
                fill=tuple(int(v * fade) for v in (235, 242, 255)), anchor="mm")
        img[944:984, 760:1160] = np.asarray(pil)
    if t >= T_TITLE:
        k = smooth((t - T_TITLE) / 0.6)
        rgb, al = TITLE_PILL
        sc = 0.5
        y = 14 + (1 - ease_out_back((t - T_TITLE) / 0.7)) * -50
        blit(img, TITLE_PILL, W / 2 - rgb.shape[1] * sc / 2, y, sc, k)


def full_render(out):
    tmp = out.replace(".mp4", ".part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
           "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for f in range(n):
        p.stdin.write(np.ascontiguousarray(render_frame(f / FPS)).tobytes())
        if f % 60 == 0:
            print(f"frame {f}/{n}", flush=True)
    p.stdin.close()
    if p.wait() == 0:
        os.replace(tmp, out)
        print("wrote", out)
    else:
        sys.exit("ffmpeg failed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        os.makedirs("preview", exist_ok=True)
        for ts in sys.argv[2:]:
            im = render_frame(float(ts))
            Image.fromarray(im).save(f"preview/t{float(ts):05.2f}.png")
    else:
        full_render(sys.argv[1] if len(sys.argv) > 1 else "flight-globe.mp4")
