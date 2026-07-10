import pytest

from shooklink.input.events import (
    KeyAction,
    KeyEvent,
    KeyLocation,
    Modifiers,
    MouseButton,
    MouseButtonEvent,
    PointerPositionEvent,
    WheelEvent,
)
from shooklink.input.macos_backend import mac_keycode_to_usage
from shooklink.input.windows_backend import windows_key_to_usage


@pytest.mark.parametrize(
    ("mac_keycode", "virtual_key", "expected_usage"),
    [
        (0, 0x41, 0x04),
        (18, 0x31, 0x1E),
        (29, 0x30, 0x27),
        (44, 0xBF, 0x38),
        (47, 0xBE, 0x37),
        (24, 0xBB, 0x2E),
        (27, 0xBD, 0x2D),
        (57, 0x14, 0x39),
        (56, 0xA0, 0xE1),
        (60, 0xA1, 0xE5),
        (59, 0xA2, 0xE0),
        (62, 0xA3, 0xE4),
        (54, 0x5C, 0xE7),
        (122, 0x70, 0x3A),
    ],
)
def test_macos_and_windows_keys_map_to_same_usb_usage(
    mac_keycode,
    virtual_key,
    expected_usage,
):
    assert mac_keycode_to_usage(mac_keycode) == expected_usage
    assert windows_key_to_usage(virtual_key, scan_code=0, extended=False) == expected_usage


def test_key_event_preserves_physical_and_text_fields():
    event = KeyEvent(
        action=KeyAction.DOWN,
        usage=0x38,
        scan_code=0x35,
        virtual_key=0xBF,
        text="?",
        modifiers=Modifiers.SHIFT,
        location=KeyLocation.STANDARD,
        repeat=True,
    )

    assert event.usage == 0x38
    assert event.text == "?"
    assert event.repeat is True


@pytest.mark.parametrize(
    ("mac_keycode", "virtual_key", "usage"),
    [
        (18, 0x31, 0x1E),
        (19, 0x32, 0x1F),
        (20, 0x33, 0x20),
        (21, 0x34, 0x21),
        (23, 0x35, 0x22),
        (22, 0x36, 0x23),
        (26, 0x37, 0x24),
        (28, 0x38, 0x25),
        (25, 0x39, 0x26),
        (29, 0x30, 0x27),
        (27, 0xBD, 0x2D),
        (24, 0xBB, 0x2E),
        (33, 0xDB, 0x2F),
        (30, 0xDD, 0x30),
        (42, 0xDC, 0x31),
        (41, 0xBA, 0x33),
        (39, 0xDE, 0x34),
        (50, 0xC0, 0x35),
        (43, 0xBC, 0x36),
        (47, 0xBE, 0x37),
        (44, 0xBF, 0x38),
    ],
)
def test_number_row_and_punctuation_matrix(mac_keycode, virtual_key, usage):
    assert mac_keycode_to_usage(mac_keycode) == usage
    assert windows_key_to_usage(virtual_key, 0, False) == usage


@pytest.mark.parametrize(
    ("mac_keycode", "virtual_key", "usage", "extended"),
    [
        (82, 0x60, 0x62, False),
        (83, 0x61, 0x59, False),
        (84, 0x62, 0x5A, False),
        (85, 0x63, 0x5B, False),
        (86, 0x64, 0x5C, False),
        (87, 0x65, 0x5D, False),
        (88, 0x66, 0x5E, False),
        (89, 0x67, 0x5F, False),
        (91, 0x68, 0x60, False),
        (92, 0x69, 0x61, False),
        (67, 0x6A, 0x55, False),
        (69, 0x6B, 0x57, False),
        (78, 0x6D, 0x56, False),
        (75, 0x6F, 0x54, False),
        (65, 0x6E, 0x63, False),
        (76, 0x0D, 0x58, True),
    ],
)
def test_keypad_matrix(mac_keycode, virtual_key, usage, extended):
    assert mac_keycode_to_usage(mac_keycode) == usage
    assert windows_key_to_usage(virtual_key, 0, extended) == usage


@pytest.mark.parametrize(
    ("virtual_key", "scan_code", "extended", "usage"),
    [
        (0x23, 0x4F, False, 0x59),  # Keypad 1 reports VK_END with Num Lock off.
        (0x23, 0x4F, True, 0x4D),
        (0x11, 0x1D, False, 0xE0),
        (0x11, 0x1D, True, 0xE4),
        (0x10, 0x2A, False, 0xE1),
        (0x10, 0x36, False, 0xE5),
        (0x00, 0x35, False, 0x38),
    ],
)
def test_windows_scan_code_preserves_physical_key_identity(
    virtual_key,
    scan_code,
    extended,
    usage,
):
    assert windows_key_to_usage(virtual_key, scan_code, extended) == usage


def test_keypad_and_pointer_events_are_explicit():
    keypad = KeyEvent(
        KeyAction.DOWN,
        usage=0x59,
        scan_code=0x4F,
        virtual_key=0x61,
        text="1",
        location=KeyLocation.NUMPAD,
    )
    button = MouseButtonEvent(MouseButton.X1, KeyAction.DOWN)
    position = PointerPositionEvent(-100, 250)
    wheel = WheelEvent(2, -5)

    assert keypad.location is KeyLocation.NUMPAD
    assert button.button is MouseButton.X1
    assert position.position == (-100, 250)
    assert (wheel.dx, wheel.dy) == (2, -5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"usage": -1},
        {"usage": 1 << 16},
        {"scan_code": -1},
        {"virtual_key": -1},
        {"text": "x" * 33},
        {"repeat": 1},
    ],
)
def test_invalid_key_event_fields_are_rejected(kwargs):
    values = {
        "action": KeyAction.DOWN,
        "usage": 4,
        "scan_code": 0,
        "virtual_key": 0,
        "text": "a",
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        KeyEvent(**values)
