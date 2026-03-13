# Serial Text Chat v1

Simple desktop chat for two machines connected through a null modem serial cable.

## Features

- Connect to a serial port with selectable or manually entered baud rates
- Send and receive multiline UTF-8 text messages
- Use a tall editor-style input box with its own scrollbar
- Send plain text or obfuscated text without JSON or sender metadata
- Copy the last received raw data or the decoded version to the clipboard
- Show the raw sent and received data in the log
- Send files by file picker or drag and drop
- Save received files into `received_files/`
- Refresh the available serial port list

## Requirements

- Python 3.10+
- `pyserial`
- `tkinterdnd2`

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
4. Set the same baud rate on both machines.
5. Click `Connect`.
6. Type your message in the large input box.
7. Click `Send Plain` to transmit the message body as-is.
8. Click `Send Encoded` to transmit the message body after custom base64-plus-marker obfuscation.
9. Click `Select Files` to choose files and send them.
10. Or drag files into the drop area to send them.
11. Use `Copy to Clipboard` to copy the last received raw data.
12. Use `Copy to Clipboard After Decode` to decode the last received data and copy the result.

Keyboard shortcuts:

- `Ctrl+Enter`: send plain
- `Ctrl+Shift+Enter`: send encoded

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

- Messages are framed with a `NUL` terminator so the full body can be received as one unit.
- File transfers are sent as chunked control frames over the same serial link.
- You can type a port manually if it does not appear in the list.
- `loop://` also works for a local self-test on one machine.
- Plain messages send only the original body. Encoded messages send only the obfuscated body.
- Encoded messages insert `A`, `B`, `C` markers after each 4-character base64 block.
- Received files are saved under `received_files/` next to the program.
- Common higher baud rates like `230400`, `460800`, `921600`, and `1000000` are listed, and you can type other values manually.
- If one direction starts corrupting at very high baud rates, try `460800` or `921600` first. This build adds a small send delay above `460800` to improve stability.
