#!/usr/bin/python3

#####################################
### @file SparsityPlotter.py
### @author Paul Daum
### @date 18.10.2018
### @brief This file contains the
### sparsity plotter class.
#####################################

# For plotting
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# Here, CasADi is used for its symbolic
# semantics
import casadi as cas
import numpy as np

##
# @class SparsityPlotter
# @brief This class provides an interface
# to create plots and figures of the 
# sparsity structure of a given symbolic
# matrix.
# TODO: For matrices whose nonzeroes have
# numerical values (i.e. the matrices have
# been evaluated), it can also color-code
# the magnitudes of the corresponding cells
# TODO: It also provides a callback method
# to update the plot of the sparsity structure 
# and the value magnitude while iterating
##
class SparsityPlotter:

  ##
  # @brief Initializes the sparsity plotter
  # @param rows The number of rows in the sparsity pattern
  # @param cols The number of columns in the sparsity pattern
  ##
  def __init__(self, fignum, rows, cols, gridSpacing = 1):
    # Fetch rows and columns
    self.rows = rows
    self.cols = cols

    # Create a figure handle
    self.figure = plt.figure(fignum)
    self.ax = self.figure.gca()

    # Set the grid-spacings
    major_ticks_rows = np.arange(0,rows,gridSpacing)
    minor_ticks_rows = np.arange(0,rows,1)
    major_ticks_cols = np.arange(0,cols,gridSpacing)
    minor_ticks_cols = np.arange(0,cols,1)

    self.ax.set_xticks(major_ticks_cols)
    self.ax.set_xticks(minor_ticks_cols, minor=True)
    self.ax.set_yticks(major_ticks_rows)
    self.ax.set_yticks(minor_ticks_rows, minor=True)

    # Configure the plot
    self.ax.grid(True)
    self.ax.xaxis.tick_top()
    self.ax.yaxis.tick_left()
    self.ax.set_ylim([self.rows, 0])
    self.ax.set_xlim([0, self.cols])
    self.ax.set_aspect('equal')

    # Create color palette
    colors_rgb256 = [
      (204, 12, 12), # Light red
      (190, 190, 190), # Gray
      (12, 0, 255) # Blue
    ]

    # Normalize all rgb elements to the interval [0,1]
    colors = [ tuple(val / 255.0 for val in rgb) for rgb in colors_rgb256 ]
    
    # Create a color map
    self.colormap = LinearSegmentedColormap.from_list(
      'Colormap',
      colors,
      N = 100
    )


  ## 
  # @brief Shows a plot of the sparsity pattern
  # @param matrixSX The matrix to be analyzed
  ##
  def printSparsity(self, matrix):
    # Check if matrix has correct dimensions
    if(matrix.rows() != self.rows or 
       matrix.columns() != self.cols):
       raise ValueError("Given matrix has incompatible dimension!")

    # Clear the plot
    [patch.remove() for patch in self.ax.patches]
    #self.ax.clear()

    # If the matrix is DM, we can color code the cells
    if type(matrix) == cas.DM:
      # Compute the average of all the cells
      mean = np.mean(matrix.elements())
      
    # Iterate over cells, create a box if there's
    # a nonzero
    boxes = []
    for i in range(self.rows):
      for j in range(self.cols):
        if matrix.has_nz(i,j):
          # Create a box for the cell
          box = Rectangle((i,j),1,1, edgecolor='white')
          
          # If the matrix is numeric, color code the cells
          if type(matrix) == cas.DM:
            color = self.createCellColor(mean, matrix[i,j].full())
            if type(color) != tuple:
              # This is a hack. Sometimes the color doesn't come
              # out as a tuple
              color = color.flatten()
              
            box.set_facecolor(color)

          # Add the box to the plot
          self.ax.add_patch(box)
          
    # Return the figure handle
    return self.figure
    

  ##
  # @brief Compute the rgb color that corresponds
  #        to a distance between two values.
  # @param mean Mean value
  # @param dval A double value
  # @return A color values (rgb)
  ##
  def createCellColor(self, mean, dval):
    # Get the magnitude of the difference
    mean_exp = np.floor(np.log10(np.abs(mean)))
    dval_exp = np.floor(np.log10(np.abs(dval)))
    diff_exp = mean_exp - dval_exp

    # Set limits
    max_exp = 16.0
    min_exp = -16.0
    
    # Normalize the exponent
    exp_normalized = diff_exp / max_exp

    # Initialize the color selector
    alpha = 0.5

    # Compute color selector by mapping
    # the exponent from [min_exp, max_exp]
    # to [0, 1]
    # exp = min_exp -> alpha = 0.0
    # exp = 0 -> alpha = 0.5
    # exp = max_exp -> alpha = 1.0
    scale = max_exp - min_exp
    alpha = (diff_exp + max_exp) / scale

    # Clip color selector
    if diff_exp > max_exp:
      alpha = 1.0
    elif diff_exp < min_exp:
      alpha = 0.0

    return self.colormap(alpha)


  ##
  # @brief TODO
  ##
  def updatePlot(self):
    pass



###########################################################################
###                                                                     ###
###                    END OF SPARSITYPLOTTER CLASS                     ###
###                                                                     ###
###########################################################################

# Unit test
if __name__ == '__main__':
  import time

  # Create a test matrix
  n = 5
  smat = cas.SX.eye(n)
  smat[0,1] = 1.0
  smat[3,2] = 1.0

  # Create a sparsity plotter
  plotter = SparsityPlotter(1,n,n,1)
  #plotter.printSparsity(smat)

  dmat = cas.DM.eye(n)
  dmat[0,1] = 1e-3
  dmat[3,2] = 1e6
  dmat[4,2] = 1e-12
  figure1 = plotter.printSparsity(dmat)
  plt.show(block=False)
  
  # Alter the matrix after some time and replot
  time.sleep(1)
  dmat[3,0] = 1e12
  figure2 = plotter.printSparsity(dmat)

  plt.show()
  print("c")