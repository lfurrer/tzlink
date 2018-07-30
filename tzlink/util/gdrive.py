#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Google drive doesn't do virus checking for large files,
therefore it wants a confirmation from the user.
The confirmation uses a session cookie, so two GET requests
are needed.

Based on https://stackoverflow.com/a/39225039
'''


import sys
import time

import requests


CHUNK_SIZE = 2**15
URL = "https://docs.google.com/uc?export=download"


def main():
    '''
    Run as script: read ID from ARGV, write downloaded file to STDOUT.
    '''
    try:
        id_ = sys.argv[1]
    except IndexError:
        sys.exit('file ID needed as command-line arg')
    download_file_from_google_drive(id_, sys.stdout.buffer)


def download_file_from_google_drive(id_, destination):
    '''
    Use two GET requests to download a file from Google drive.
    '''
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def _save_response_content(response, destination):
        progbar = ProgressBar('Download from Google Drive',
                              int(response.headers.get('Content-Length', 0)))
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive chunks
                destination.write(chunk)
            progbar.update(len(chunk))
        progbar.close()

    session = requests.Session()

    response = session.get(URL, params={'id': id_}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id_, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


class ProgressBar:
    '''
    A terminal-based progress bar for network trasmission.
    '''
    def __init__(self, name, total_size, out_stream=sys.stderr,
                 barwidth=30, interval=1):
        self.name = name
        self.out = out_stream
        self.barwidth = barwidth
        self.interval = interval

        self.transmitted = 0
        self.cached = 0
        self.total = total_size
        self.start = time.time()
        self.last = self.start

    def update(self, size):
        '''Update the progress bar.'''
        now = time.time()
        self.cached += size
        if now-self.last >= self.interval:
            self._show_update(now)

    def close(self):
        '''Final line.'''
        self._show_update(time.time(), total_speed=True)
        self.out.write('\n')

    def _show_update(self, now, total_speed=False):
        duration = now-self.start
        self.transmitted += self.cached
        if total_speed:
            speed = self.transmitted/duration
        else:
            speed = self.cached/(now-self.last)
        self.last = now
        self.cached = 0
        self._format(duration, speed)

    def _format(self, duration, speed):
        if self.total:
            width = round(self.transmitted*self.barwidth/self.total)
            bar_ = ('#'*width).ljust(self.barwidth)
        else:
            bar_ = '???'.center(self.barwidth, '-')
        transmitted = self.size_fmt(self.transmitted)
        speed = self.size_fmt(speed)
        duration = self.time_fmt(duration)
        line = self.template.format(name=self.name,
                                    bar=bar_,
                                    done=transmitted,
                                    speed=speed,
                                    duration=duration)
        self.out.write(line)

    template = '\r{name}  [{bar}]  {done}  {speed}/s  {duration}'

    @staticmethod
    def size_fmt(num):
        '''Convert file size to a human-readable format.'''
        for prefix in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
            if abs(num) < 1000:
                # Use 1000 instead of 1024 to avoid 4-digit numbers
                # (the .3g format produces E notation for them).
                break
            num /= 1024
        else:
            prefix = 'Yi'
        return '{:.3g} {}B'.format(num, prefix).rjust(9)

    @staticmethod
    def time_fmt(seconds):
        '''Convert seconds to one of three formats.'''
        if seconds < 1:
            return '{:.3g}'.format(seconds)
        minutes, seconds = divmod(round(seconds), 60)
        if minutes < 60:
            return '{:02}:{:02}'.format(minutes, seconds)
        hours, minutes = divmod(minutes, 60)
        return '{}:{:02}:{:02}'.format(hours, minutes, seconds)


if __name__ == "__main__":
    main()
