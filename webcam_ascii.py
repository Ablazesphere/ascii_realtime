#!/usr/bin/env python3

import os
import sys
import time
import cv2
import numpy as np
import argparse
import signal
import curses
from typing import List, Dict, Any, Optional, Tuple


class StyledWebcamAsciiAnimator:
    
    CHAR_SETS = {
        'standard': '@%#*+=-:. ',
        'detailed': '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\'. ',
        'blocks': '█▓▒░ ',
        'simple': '#. ',
        'matrix': '@MATRIX$ ',
        'binary': '10 ',
        'braille': '⣿⣷⣯⣟⡿⢿⣻⣽⣾ '
    }
    
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
    }
    
    def __init__(self, options=None):
        self.options = {
            'camera_id': 0,
            'width': 80,
            'height': 24,
            'fps': 15,
            'brightness': 0,
            'contrast': 1.0,
            'color_mode': 'none',
            'charset': 'standard',
            'invert': False,
            'flip_horizontal': True,
            'show_fps': False,
            'fps_position': 'corner',  # Options: 'corner', 'header', 'bottom_right', 'none'
            'line_numbers': True,
            'header': True,
            'fill_char': '0',
            'padding_char': ' ',
            'style': 'default',
        }
        
        if options:
            self.options.update(options)
        
        self.cap = None
        self.running = False
        self.chars = self.get_charset()
        self.min_frame_time = 1.0 / self.options['fps'] if self.options['fps'] > 0 else 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps_start_time = 0
        self.fps = 0
        self.term_width, self.term_height = self.get_terminal_size()
        self.line_prefix = 'K' if self.options['style'] == 'default' else ''
        self.line_suffix = '$' if self.options['style'] == 'default' else ''
        self.stdscr = None
        if self.options['style'] == 'curses':
            self.init_curses()
    
    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.curs_set(0)
        self.stdscr.clear()
    
    def cleanup_curses(self):
        if self.stdscr:
            curses.endwin()
    
    def get_charset(self):
        charset = self.options['charset']
        if charset in self.CHAR_SETS:
            chars = self.CHAR_SETS[charset]
        else:
            chars = self.CHAR_SETS['standard']
        
        if self.options['invert']:
            return chars[::-1]
        return chars
    
    def get_terminal_size(self):
        try:
            columns, lines = os.get_terminal_size()
            return columns, lines
        except (AttributeError, OSError):
            return 100, 30
    
    def start_webcam(self):
        try:
            self.cap = cv2.VideoCapture(self.options['camera_id'])
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.options['camera_id']}")
                return False
            
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.running = True
            
            return True
        except Exception as e:
            print(f"Error starting webcam: {str(e)}")
            return False
    
    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        self.cleanup_curses()
    
    def adjust_image(self, frame):
        adjusted = frame.copy().astype(np.float32)
        adjusted = adjusted * self.options['contrast']
        adjusted = adjusted + self.options['brightness']
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def resize_frame(self, frame):
        aspect_ratio_correction = 0.5
        target_height = int(self.options['height'] / aspect_ratio_correction)
        return cv2.resize(frame, (self.options['width'], target_height))
    
    def frame_to_ascii(self, frame) -> List[str]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_chars = len(self.chars)
        char_indices = (gray / 255.0 * (num_chars - 1)).astype(np.int32)
        
        lines = []
        for row in range(gray.shape[0]):
            line = self.line_prefix
            
            for col in range(gray.shape[1]):
                char_idx = char_indices[row, col]
                ascii_char = self.chars[char_idx]
                
                if self.options['color_mode'] == 'none':
                    line += ascii_char
                elif self.options['color_mode'] == 'green':
                    line += f"{self.COLORS['green']}{ascii_char}{self.COLORS['reset']}"
                elif self.options['color_mode'] == 'yellow':
                    line += f"{self.COLORS['yellow']}{ascii_char}{self.COLORS['reset']}"
                elif self.options['color_mode'] == 'blue':
                    line += f"{self.COLORS['blue']}{ascii_char}{self.COLORS['reset']}"
                elif self.options['color_mode'] == 'rainbow':
                    hue = (col / gray.shape[1]) * 360
                    r, g, b = self.hsv_to_rgb(hue, 1.0, 0.8)
                    line += f"\033[38;2;{r};{g};{b}m{ascii_char}\033[0m"
                elif self.options['color_mode'] == 'ansi':
                    b, g, r = frame[row, col]
                    line += f"\033[38;2;{r};{g};{b}m{ascii_char}\033[0m"
                elif self.options['color_mode'] == 'ansi_bg':
                    b, g, r = frame[row, col]
                    line += f"\033[48;2;{r};{g};{b}m{ascii_char}\033[0m"
            
            line += self.line_suffix
            lines.append(line)
        
        return lines
    
    def hsv_to_rgb(self, h, s, v):
        h = (h % 360) / 60.0
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return int(r * 255), int(g * 255), int(b * 255)
    
    def clear_screen(self):
        if self.stdscr:
            self.stdscr.clear()
        else:
            if os.name == 'nt':
                os.system('cls')
            else:
                sys.stdout.write('\033[2J\033[H')
                sys.stdout.flush()
    
    def calculate_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        if self.frame_count > 100:
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def should_process_frame(self):
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        if elapsed < self.min_frame_time:
            time.sleep(self.min_frame_time - elapsed)
            return False
            
        self.last_frame_time = current_time
        return True
    
    def create_header(self, width):
        header = ""
        
        if self.options['line_numbers']:
            header += "    "
        
        for i in range(10):
            marker = str(i)
            dots = "." * 9
            
            if self.options['color_mode'] == 'none':
                header += f"{marker}{dots}"
            else:
                header += f"{self.COLORS['bright_yellow']}{marker}{dots}{self.COLORS['reset']}"
        
        if self.options['show_fps'] and self.options['fps_position'] == 'header':
            fps_text = f"FPS: {self.fps:.1f}"
            available_width = width - len(fps_text)
            header = header[:available_width] + fps_text
        else:
            header = header[:width]
            
        return header
    
    def format_ascii_with_line_numbers(self, ascii_lines):
        formatted_lines = []
        
        if self.options['header']:
            header = self.create_header(self.options['width'] + 4)
            formatted_lines.append(header)
        
        for i, line in enumerate(ascii_lines):
            line_num = str(i).zfill(2)
            
            if self.options['color_mode'] == 'none':
                formatted_line = f"{line_num}: {line}"
            else:
                formatted_line = f"{self.COLORS['bright_yellow']}{line_num}{self.COLORS['reset']}: {line}"
            
            formatted_lines.append(formatted_line)
        
        return formatted_lines
    
    def generate_curses_output(self, ascii_lines):
        if not self.stdscr:
            return
        
        max_y, max_x = self.stdscr.getmaxyx()
        
        if self.options['header']:
            header = self.create_header(min(self.options['width'] + 4, max_x - 1))
            self.stdscr.addstr(0, 0, header)
        
        y_offset = 1 if self.options['header'] else 0
        for i, line in enumerate(ascii_lines):
            if y_offset + i >= max_y:
                break
                
            line_num = str(i).zfill(2)
            self.stdscr.addstr(y_offset + i, 0, f"{line_num}: ", curses.A_BOLD)
            self.stdscr.addstr(y_offset + i, 4, line)
        
        if self.options['show_fps']:
            fps_text = f"FPS: {self.fps:.1f}"
            
            if self.options['fps_position'] == 'corner':
                try:
                    self.stdscr.addstr(0, max_x - len(fps_text) - 1, fps_text)
                except:
                    pass
            elif self.options['fps_position'] == 'bottom_right':
                try:
                    self.stdscr.addstr(min(y_offset + len(ascii_lines), max_y - 1), 
                                      max_x - len(fps_text) - 1, 
                                      fps_text)
                except:
                    pass
            
        self.stdscr.refresh()
    
    def get_fps_display(self):
        fps_text = f"FPS: {self.fps:.1f}"
        
        if self.options['color_mode'] != 'none':
            return f"{self.COLORS['bright_cyan']}{fps_text}{self.COLORS['reset']}"
        return fps_text
    
    def run(self):
        if not self.start_webcam():
            return
        
        print("Starting Styled Webcam ASCII Animator...")
        print("Press Ctrl+C to exit")
        time.sleep(1)
        
        try:
            while self.running:
                if not self.should_process_frame():
                    continue
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Error: Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                if self.options['flip_horizontal']:
                    frame = cv2.flip(frame, 1)
                
                frame = self.adjust_image(frame)
                resized = self.resize_frame(frame)
                ascii_lines = self.frame_to_ascii(resized)
                
                if self.options['show_fps']:
                    self.calculate_fps()
                
                if self.options['line_numbers']:
                    formatted_lines = self.format_ascii_with_line_numbers(ascii_lines)
                else:
                    formatted_lines = ascii_lines
                
                if self.options['style'] == 'curses':
                    self.generate_curses_output(ascii_lines)
                else:
                    self.clear_screen()
                    
                    output = []
                    
                    if self.options['show_fps'] and self.options['fps_position'] == 'corner':
                        fps_text = self.get_fps_display()
                        if len(formatted_lines) > 0:
                            padding = max(0, self.options['width'] - len(fps_text))
                            fps_display = " " * padding + fps_text
                            output.append(fps_display)
                    
                    output.extend(formatted_lines)
                    
                    if self.options['show_fps'] and self.options['fps_position'] == 'bottom_right':
                        fps_text = self.get_fps_display()
                        padding = max(0, self.options['width'] - len(fps_text))
                        fps_display = " " * padding + fps_text
                        output.append(fps_display)
                    
                    print('\n'.join(output))
        
        except KeyboardInterrupt:
            print("\nStopping Styled Webcam ASCII Animator...")
        finally:
            self.stop_webcam()


def parse_args():
    parser = argparse.ArgumentParser(description='Styled Webcam ASCII Animator')
    
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('-W', '--width', type=int, default=80,
                        help='Width of the ASCII output (default: 80)')
    parser.add_argument('-H', '--height', type=int, default=24,
                        help='Height of the ASCII output (default: 24)')
    parser.add_argument('-f', '--fps', type=int, default=15,
                        help='Maximum frames per second (default: 15)')
    parser.add_argument('-b', '--brightness', type=float, default=0,
                        help='Brightness adjustment (-128 to 128, default: 0)')
    parser.add_argument('-c', '--contrast', type=float, default=1.0,
                        help='Contrast adjustment (0 to 2.0, default: 1.0)')
    parser.add_argument('-m', '--color-mode', 
                        choices=['none', 'ansi', 'ansi_bg', 'green', 'yellow', 'blue', 'rainbow'], 
                        default='yellow',
                        help='Color mode (default: yellow)')
    parser.add_argument('-s', '--charset', 
                        choices=['standard', 'detailed', 'blocks', 'simple', 'matrix', 'binary', 'braille'], 
                        default='standard', 
                        help='Character set to use (default: standard)')
    parser.add_argument('-i', '--invert', action='store_true',
                        help='Invert the brightness of characters')
    parser.add_argument('--no-mirror', action='store_true',
                        help='Disable horizontal flipping (mirror effect)')
    parser.add_argument('--fps-display', 
                        choices=['none', 'corner', 'header', 'bottom_right'], 
                        default='none',
                        help='FPS display location (default: none)')
    parser.add_argument('--no-line-numbers', action='store_true',
                        help='Disable line numbers')
    parser.add_argument('--no-header', action='store_true',
                        help='Disable the header/ruler line')
    parser.add_argument('--fill-char', type=str, default='0',
                        help='Character to use for filling empty space (default: 0)')
    parser.add_argument('--style', choices=['default', 'plain', 'curses'], default='default',
                        help='Style of the output (default: default)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    options = {
        'camera_id': args.device,
        'width': args.width,
        'height': args.height,
        'fps': args.fps,
        'brightness': args.brightness,
        'contrast': args.contrast,
        'color_mode': args.color_mode,
        'charset': args.charset,
        'invert': args.invert,
        'flip_horizontal': not args.no_mirror,
        'show_fps': args.fps_display != 'none',
        'fps_position': args.fps_display,
        'line_numbers': not args.no_line_numbers,
        'header': not args.no_header,
        'fill_char': args.fill_char,
        'style': args.style,
    }
    
    animator = StyledWebcamAsciiAnimator(options)
    
    def signal_handler(sig, frame):
        animator.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    animator.run()


if __name__ == "__main__":
    main()