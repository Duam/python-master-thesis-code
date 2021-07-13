
import numpy as np

################################## PULSES #########################################
def pulse_single(t:float, width:float, amp:float=1.0):
  return amp if (t >= 0 and t < width) else 0.0

def pulse_periodic(t:float, T:float, width:float, amp:float=1.0):
  return pulse_single(np.mod(t,T), width, amp) if T != 0.0 else amp/2.0

def pulse_FM_linear(t:float, T:float, width:float, amp:float=1.0):
  return pulse_periodic(t, T/2**np.floor((t/float(T))), width, amp) if t >= 0 else 0.0