#!/usr/bin/python3

#####################################
### @file SpiralBuffer.py
### @author Paul Daum
### @date 12.02.2019
### @brief This file contains the
### SpiralBuffer class.
#####################################

from collections import deque


class SpiralBuffer:
  """

  """

  def __init__(self, size1, size2):

    # Check input types
    assert isinstance(size1, int)
    assert isinstance(size2, int)

    # Fetch parameters
    self._size1 = size1
    self._size2 = size2
    self._numel = size1 * size2

    # Initialize data container
    self._data = deque([0 for j in range(self._numel)], maxlen = self._numel)

    pass

  def __getitem__(self, key):

    # Check input type
    assert isinstance(key, tuple)

    # Fetch parallel indices
    k, i = key

    # Check index sizes
    assert k < self._size1
    assert i < self._size2

    # Compute the serial index
    index = i * self._size1 + k

    # Return the desired element
    return self._data[index]


  def __lshift__(self, elem):

    # Append the element to the right
    self._data.append(elem)


  def __str__(self):
    """ Creates a numpy array like string """
    s = "[ "
    for i in range(self._size2):
      s += "[ "

      for k in range(self._size1):
        s += str(self[k,i])

        if k + 1 < self._size1:
          s += ", "

      s += " ]"
      if i + 1 < self._size2:
        s += ", "
        s += "\n  "

    s += " ]"
    return s



if __name__ == '__main__':

  # Create the buffer
  spuf = SpiralBuffer(3, 2)
  print(spuf)

  # Fill it with some dummy values
  spuf << 1.0
  spuf << 2.0
  spuf << 3.0
  spuf << 4.0

  # Print out a specific element
  elem = spuf[2,0]
  print(elem)