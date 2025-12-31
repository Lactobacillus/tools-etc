import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set, Union, Optional, Literal, Any, Callable

import PyPDF2
import tiktoken

EXCLUDED_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
    '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.exe', '.dll', '.so', '.dylib',
    '.pyc', '.pyo', '.class',
    '.bin', '.dat', '.db', '.sqlite',
    '.ttf', '.otf', '.woff', '.woff2'}


def is_binary_file(
        filepath: str) -> bool:

    try:

        with open(filepath, 'rb') as fs:

            chunk = fs.read(8192)

            return b'\x00' in chunk

    except KeyboardInterrupt:

        sys.exit(0)

    except Exception:

        return True


def extract_text_from_pdf(
        filepath: str) -> str:

    text_parts = list()

    try:

        with open(filepath, 'rb') as fs:

            reader = PyPDF2.PdfReader(fs)

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text_parts.append(page_text)

    except KeyboardInterrupt:

        sys.exit(0)

    except Exception as e:

        print('[warning] Cannot read PDF file: {}'.format(filepath))
        print(e)

    return '\n'.join(text_parts)


def read_text_file(
        filepath: str) -> str:

    try:

        with open(filepath, 'r', encoding = 'UTF-8') as fs:

            return fs.read()

    except KeyboardInterrupt:

        sys.exit(0)

    except Exception as e:

        print('[warning] 파일 읽기 실패: {}'.format(filepath))
        print(e)

        return ''


def count_tokens(
        text: str,
        encoder: tiktoken.Encoding) -> int:

    if not text:

        length = 0

    else:

        length = len(encoder.encode(text))

    return length


def scan_directory(
        base_path: str,
        encoder: tiktoken.Encoding) -> dict[str, int]:

    stats = {
        'tokens': list(),
        'total_files': 0,
        'pdf_files': 0,
        'text_files': 0,
        'skipped_files': 0}

    for root, dirs, files in os.walk(base_path):

        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'venv', '.git'}]

        for filename in files:

            filepath = os.path.join(root, filename)
            _, ext = os.path.splitext(filename)
            ext = ext.lower()

            # excluded files
            if ext in EXCLUDED_EXTENSIONS:

                stats['skipped_files'] = stats['skipped_files'] + 1

                continue

            # PDF files
            if ext == '.pdf':

                text = extract_text_from_pdf(filepath)
                tokens = count_tokens(text, encoder)

                stats['tokens'].append(tokens)
                stats['pdf_files'] = stats['pdf_files'] + 1
                stats['total_files'] = stats['total_files'] + 1
                print('[info] {}: {:,} tokens'.format(filepath, tokens))

                continue

            # binary files
            if is_binary_file(filepath):

                stats['skipped_files'] = stats['skipped_files'] + 1

                continue

            # text files
            text = read_text_file(filepath)
            tokens = count_tokens(text, encoder)

            stats['tokens'].append(tokens)
            stats['text_files'] = stats['text_files'] + 1
            stats['total_files'] = stats['total_files'] + 1
            print('[info] {}: {:,} tokens'.format(filepath, tokens))

    return stats


def main(
        args: Dict[str, Any]) -> None:

    if not os.path.exists(args.base_path):

        print('[error] <base_path> does not exist: {}'.format(args.base_path))
        sys.exit(0)

    try:

        encoder = tiktoken.encoding_for_model(args.model)

    except KeyboardInterrupt:

        sys.exit(0)

    except KeyError:

        encoder = tiktoken.get_encoding('cl100k_base')

    stats = scan_directory(args.base_path, encoder)

    print('-' * 60)
    print('[info] # of total files: {:,}'.format(stats['total_files']))
    print('[info] # of PDF files: {:,}'.format(stats['pdf_files']))
    print('[info] # of text files: {:,}'.format(stats['text_files']))
    print('[info] # of skipped files: {:,}'.format(stats['skipped_files']))
    print('[info] # of total tokens: {:,}'.format(np.sum(stats['tokens'])))
    print('[info] Avg. tokens per documents: {:,}'.format(np.mean(stats['tokens'])))
    print('[info] Med. tokens per documents: {:,}'.format(np.median(stats['tokens'])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type = str, required = True,
                        help = 'Base directory')
    parser.add_argument('--model', type = str,
                        default = 'gpt-4o',
                        help = 'Tokenizer model (default: gpt-4o)')
    args = parser.parse_args()

    main(args)
