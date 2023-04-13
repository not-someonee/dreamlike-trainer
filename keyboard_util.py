import time
import atexit
import termios
import sys
import os

old_settings = None

def enable():
  global old_settings
  old_settings = termios.tcgetattr(sys.stdin)
  new_settings = termios.tcgetattr(sys.stdin)
  new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)
  new_settings[6][termios.VMIN] = 0
  new_settings[6][termios.VTIME] = 0
  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)


@atexit.register
def disable():
  global old_settings
  if old_settings:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def get_pressed_keys():
  keys = []
  ch = os.read(sys.stdin.fileno(), 1)
  while ch is not None and len(ch) > 0:
    keys.append(chr(ch[0]))
    ch = os.read(sys.stdin.fileno(), 1)
  return keys


enable()
