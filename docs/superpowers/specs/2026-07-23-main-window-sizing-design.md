# Main Window Sizing Design

## Goal

Prevent the ShookLink main-window controls from overlapping or being clipped at
startup while preserving access on displays whose available height is smaller
than the full form.

## Chosen Design

- Increase the initial main-window size from `920x760` to `1100x1000`.
- Put the existing central form inside a borderless, widget-resizable
  `QScrollArea`.
- Keep the form's current spacing and editor minimum heights.
- Allow vertical scrolling when the available window height is below the form's
  layout minimum; avoid a horizontal scrollbar at the new default width.
- Preserve all existing widget references, signals, shortcuts, drag-and-drop,
  and visual styling.

## Alternatives Rejected

- Increasing only the fixed minimum size would still clip the form on a 1080p
  desktop after accounting for the title bar and taskbar or Dock.
- Compressing editor heights and panel spacing would make the text and terminal
  workflow less usable and would only defer the layout problem as features are
  added.

## Verification

- Assert the default window is `1100x1000` or larger after construction.
- Assert the central widget is a resizable `QScrollArea` with the existing form
  as its content.
- Resize below the form minimum and verify vertical scrolling is available and
  key controls retain non-overlapping geometry.
- Run the complete UI and project test suites plus an offscreen startup smoke
  test.
