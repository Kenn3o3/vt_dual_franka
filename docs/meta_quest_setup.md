#!/bin/bash
adb reverse tcp:8082 tcp:8082
adb shell settings put global stay_on_while_plugged_in 3
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close --ei timeout 0
adb shell setprop debug.oculus.guardian_pause 1
echo "Quest USB setup done"

127.0.0.1