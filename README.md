# Styled Webcam ASCII Animator

A Python tool that transforms your webcam feed into real-time ASCII art in your terminal with various styling options.

## Features

- **Live ASCII Conversion**: Transform your webcam feed into ASCII characters in real-time
- **Multiple Character Sets**: Choose from various ASCII character sets including standard, detailed, blocks, matrix, binary, and braille
- **Color Modes**: Display in plain text or with various color options including ANSI colors, green (Matrix-style), yellow, blue, and rainbow
- **Performance Monitoring**: Optional FPS counter with customizable position (corner, header, or bottom right)
- **Styling Options**: Adjustable contrast, brightness, and image inversion
- **Terminal Integration**: Choose between default, plain, or curses-based display styles
- **UI Elements**: Optional header/ruler and line numbers for easier viewing

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Working webcam

## Installation

1. Clone this repository or download the script.

2. Install the required dependencies:

```bash
pip install opencv-python numpy
```

3. Make the script executable (Linux/Mac):

```bash
chmod +x webcam_ascii.py
```

## Usage

### Basic Usage

```bash
python webcam_ascii.py
```

### With FPS Counter in Bottom Right

```bash
python webcam_ascii.py --fps-display bottom_right
```

### Full Example with Custom Options

```bash
python webcam_ascii.py --fps-display bottom_right --color-mode green --charset blocks --width 100 --height 30 --contrast 1.2 --brightness 10
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-d`, `--device` | Camera device ID (default: 0) |
| `-W`, `--width` | Width of the ASCII output (default: 80) |
| `-H`, `--height` | Height of the ASCII output (default: 24) |
| `-f`, `--fps` | Maximum frames per second (default: 15) |
| `-b`, `--brightness` | Brightness adjustment (-128 to 128, default: 0) |
| `-c`, `--contrast` | Contrast adjustment (0 to 2.0, default: 1.0) |
| `-m`, `--color-mode` | Color mode: none, ansi, ansi_bg, green, yellow, blue, rainbow (default: yellow) |
| `-s`, `--charset` | Character set: standard, detailed, blocks, simple, matrix, binary, braille (default: standard) |
| `-i`, `--invert` | Invert the brightness of characters |
| `--no-mirror` | Disable horizontal flipping (mirror effect) |
| `--fps-display` | FPS display location: none, corner, header, bottom_right (default: none) |
| `--no-line-numbers` | Disable line numbers |
| `--no-header` | Disable the header/ruler line |
| `--fill-char` | Character to use for filling empty space (default: 0) |
| `--style` | Style of the output: default, plain, curses (default: default) |

## Character Sets

- **standard**: `@%#*+=-:. `
- **detailed**: `$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^'. `
- **blocks**: `█▓▒░ `
- **simple**: `#. `
- **matrix**: `@MATRIX$ `
- **binary**: `10 `
- **braille**: `⣿⣷⣯⣟⡿⢿⣻⣽⣾ `

## Control

Press Ctrl+C to exit the program.

## Examples

### Matrix Style

```bash
python webcam_ascii.py --color-mode green --charset matrix --fps-display bottom_right
```

### High Detail Monochrome

```bash
python webcam_ascii.py --color-mode none --charset detailed --width 120 --height 40
```

### Rainbow Block Art

```bash
python webcam_ascii.py --color-mode rainbow --charset blocks --fps-display bottom_right
```

## Troubleshooting

- If you get a camera error, try changing the device ID with `--device 1` (or another number)
- For performance issues, try reducing the width and height or lowering the FPS
- If characters appear stretched, adjust the width to match your terminal dimensions

Thanks Claude :))
