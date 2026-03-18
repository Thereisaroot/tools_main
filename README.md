# Serial Text Chat v1

Simple desktop serial app for two machines connected through a null modem cable.

## Features

- Connect to a serial port with selectable or manually entered baud rates
- Send and receive multiline UTF-8 text messages
- Use a tall editor-style input box with its own scrollbar
- Send plain text or obfuscated text without JSON or sender metadata
- Copy the last received raw data or the decoded version to the clipboard
- Send files by file picker or drag and drop
- Save received files into `received_files/`
- Open the download folder from the app
- Toggle serial-based keyboard/mouse sharing between macOS and Windows
- Share keyboard keys, relative mouse movement, mouse buttons, and mouse wheel
- Optionally auto-toggle remote control from a configured whole-desktop screen edge

## Requirements

- Python 3.10+
- `pyserial`
- `tkinterdnd2`
- `pynput`

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

## Text and File Use

1. Connect the two machines with a null modem serial cable.
2. Start the program on both machines.
3. Select the correct serial port on each machine.
4. Set the same baud rate on both machines.
5. Click `Connect`.
6. Type in the large message box and use `Send Plain` or `Send Encoded`.
7. Use `Select Files` or drag files into the drop area to send them.
8. Received files are written into `received_files/`.
9. `Copy to Clipboard` copies the last raw text payload.
10. `Copy to Clipboard After Decode` decodes the last obfuscated payload and copies it.

Keyboard shortcuts:

- `Ctrl+Enter`: send plain text
- `Ctrl+Shift+Enter`: send encoded text

## Input Share Use

Input sharing is `serial-only`, `hotkey-toggle`, and `bidirectional`.

- Windows hotkey: `Scroll Lock`
- macOS hotkey: `F8`
- macOS also accepts `Shift+F8` and `Ctrl+Shift+F8` because the toggle is triggered by the same `F8` key event
- Emergency stop on both platforms: `Ctrl+Alt+Shift+Backspace`
- Emergency exit on both platforms: `Ctrl+Alt+Shift+Esc`

How it works:

1. Connect both machines first.
2. Make sure both machines run this updated version.
3. On the machine that should control the other one, click `Toggle Remote Control` or press the hotkey. The global capture hooks initialize at that moment, while the receive side is prepared after serial connect.
4. When the state changes to `Controlling remote`, local keyboard and mouse events are sent over serial.
5. Press the same hotkey again, or click `Stop Remote Control`, to release control.
6. Optional: enable `Auto edge toggle`, choose the peer side, then push against that outer screen edge for about 0.5 seconds to start remote control.
7. While controlling remote, push against the opposite direction for about 0.5 seconds to stop remote control.
8. If remote control ever gets stuck, use the emergency stop or emergency exit combo locally.

Current v1 scope:

- Supported targets: `macOS <-> Windows`
- Shared input: keyboard, mouse move, click, scroll
- Mouse mode: relative movement
- Receiver policy: always armed
- Auto edge toggle: whole-desktop outer edges with configurable peer side

Not included in v1:

- clipboard sync
- complex multi-host or multi-edge topologies
- auth/encryption
- drag/file handoff between machines

## Interaction Rules

- Text messages use separate internal frames, so text send can coexist with remote control traffic.
- File send stays disabled while remote control is active.
- While file transfer is active, remote control cannot be started.
- Auto edge toggle only starts from `Idle` and only stops from `Controlling remote`.
- `Open Download Folder` stays available all the time.
- File transfer still uses per-chunk acknowledgements.
- Remote control uses a session-based `INPUT_*` control protocol on the same serial link.

## macOS Permissions

macOS global keyboard and mouse hooks need system permissions.

If the app shows `Permission required`, allow the Python app or terminal app in:

- `System Settings -> Privacy & Security -> Accessibility`
- `System Settings -> Privacy & Security -> Input Monitoring`

Then fully quit and relaunch the program.

## Port Examples

- macOS: `/dev/tty.usbserial-xxxx`, `/dev/cu.usbserial-xxxx`
- Linux USB serial: `/dev/ttyUSB0`
- Linux onboard serial: `/dev/ttyS0`
- Windows: `COM3`

## Serial Settings

The program uses these defaults:

- 8 data bits
- no parity
- 1 stop bit
- no flow control
- `NUL` terminator for payload framing

Both machines must use matching serial settings.

## Linux Permissions

If the serial port cannot be opened on Linux, add your user to the serial access group and log in again:

```bash
sudo usermod -a -G dialout "$USER"
```

## Notes

- Text payloads still send only the message body.
- Encoded text still uses the existing base64-plus-marker obfuscation.
- File transfers are chunked control frames over the same serial link.
- Input sharing uses `INPUT_START / INPUT_ACK / INPUT_BUSY / INPUT_STOP / INPUT_RELEASE_ALL` and per-event input frames.
- Both sides must be updated to the same build for input sharing and file transfer.
- Auto edge toggle settings persist between launches.
- Common higher baud rates like `230400`, `460800`, `921600`, and `1000000` are listed, and you can type other values manually.
- If one direction starts corrupting at very high baud rates, try `460800` or `921600` first. This build still adds a small send delay above `460800` to improve stability.
