'''
    Copyright (C) 2026  Geert Schulpen, Isabela Chirila-Rus

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
'''


__version__ = '0.0.2'

from bijelanalysis.TIF_Viewer import ImageViewer
from bijelanalysis.surface.bijels_area_gui import BijelsAreaApp
from bijelanalysis.edge.bijel_edges_gui import BijelsEdgesApp

#make visible for 'from imageAnalysis import *'
__all__ = [
    'ImageViewer',
	'BijelsAreaApp',
	'BijelsEdgesApp',
]
