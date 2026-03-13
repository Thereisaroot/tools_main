# Serial Text Chat v1

Simple desktop chat for two machines connected through a null modem serial cable.

## Features

- Connect to a serial port with a selectable baud rate
- Send and receive UTF-8 text messages
- Show a local chat log with timestamps
- Refresh the available serial port list

## Requirements

- Python 3.10+
- `pyserial`

## Install

```bash
python3 -m pip install -r requirements.txt
```

On some Linux distributions, `tkinter` may need a separate package:

```bash
sudo apt install python3-tk
```

## Run

```bash
python3 serial_text_chat.py
```

On macOS or Linux, you can also run:

```bash
chmod +x run_serial_text_chat.sh
./run_serial_text_chat.sh
```

On Windows, you can also run:

```bat
run_serial_text_chat.bat
```

## How to use

1. Connect the two machines with a null modem serial cable.
2. Start the program on both machines.
3. Select the correct serial port on each machine.
4. Set the same baud rate on both PCs.
5. Click `Connect`.
6. Type a message and press `Enter` or `Send`.

## Port examples

- macOS: `/dev/tty.usbserial-xxxx`, `/dev/cu.usbserial-xxxx`
- Linux USB serial: `/dev/ttyUSB0`
- Linux onboard serial: `/dev/ttyS0`
- Windows: `COM3`

## Serial settings

The program uses these defaults:

- 8 data bits
- no parity
- 1 stop bit
- no flow control
- CRLF line ending

Both PCs must use matching serial settings.

## Linux permissions

If the serial port cannot be opened on Linux, add your user to the serial access group and log in again:

```bash
sudo usermod -a -G dialout "$USER"
```

## Notes

- Messages are sent one line at a time.
- You can type a port manually if it does not appear in the list.
- `loop://` also works for a local self-test on one machine.
- This is a v1 text-only build. File transfer and binary packet support are not included yet.
