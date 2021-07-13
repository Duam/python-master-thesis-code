import time
from zerocm import ZCM
from thesis_code.zcm_message_definitions.timestamped_vector_float import timestamped_vector_float
from thesis_code.utils.zcm_constants import carousel_zcm_constants as carzcm

# Initialize ZCM
zcm = ZCM(carzcm.url)
if (not zcm.good()):
    raise Exception("ZCM not good")

# Create control messages
timestamp = int(time.time()*1e9)
control_messages = [ timestamped_vector_float() for k in range(4) ]
for msg in control_messages:
  msg.ts = timestamp
  msg.len = 1

# Set control values
control_messages[0].values = [0.375]
control_messages[1].values = [0.475]
control_messages[2].values = [0.5]
control_messages[3].values = [0.5]

""" ================================================================================================================ """

# Start zcm, let this thread sleep a while, then stop
print("Sending controls..")
zcm.start()

for k in range(len(control_messages)):
  kin_num = k + 1
  msg = control_messages[k]
  zcm.publish("VE2_KIN"+str(kin_num)+"_SET", msg)
  print("Published setpoint " + str(msg.values[0]) + " on kineos " + str(kin_num))

print("Done.")