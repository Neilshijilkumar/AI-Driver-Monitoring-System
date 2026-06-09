import winsound

alarm_on = False

def play_alarm():

    global alarm_on

    if not alarm_on:

        winsound.Beep(2500, 1000)

        alarm_on = True


def stop_alarm():

    global alarm_on

    alarm_on = False