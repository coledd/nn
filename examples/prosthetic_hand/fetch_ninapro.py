#!/usr/bin/env python3
#
# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Downloads one recording out of one Ninapro subject's zip archive without
# downloading the rest of the archive.
#
# Each Ninapro subject is published as a single zip of three MATLAB files,
# one per exercise -- ~470MB per subject for DB2, ~250MB for DB3. This
# example only needs exercise 1 (the basic hand and wrist movements), about
# a third of that. The server supports HTTP range requests, and a zip's
# table of contents sits at its end, so Python's own zipfile module can be
# pointed at a file object that fetches byte ranges on demand: it reads the
# table of contents, then streams just the one member out.
#
# Standard library only, like ../../split.py.
#
# Usage: fetch_ninapro.py <zip-url> <member-name-suffix> <output-path>

import io
import shutil
import sys
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file object over an HTTP URL."""

    def __init__(self, url):
        self.url = url
        self.pos = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            if response.headers.get("Accept-Ranges") != "bytes":
                raise IOError(f"{url} does not support range requests")
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, buffer):
        if self.pos >= self.size or len(buffer) == 0:
            return 0
        end = min(self.pos + len(buffer), self.size) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <zip-url> <member-name-suffix> <output-path>", file=sys.stderr)
        return 1
    url, suffix, output_path = sys.argv[1:]
    # The buffer size sets the size of each range request: large enough
    # that a ~100MB member takes a few dozen requests, not thousands.
    remote = io.BufferedReader(HttpRangeFile(url), buffer_size=8 << 20)
    with zipfile.ZipFile(remote) as archive:
        names = [n for n in archive.namelist() if n.endswith(suffix) and "__MACOSX" not in n]
        if len(names) != 1:
            print(f"Error: expected one member ending in {suffix} in {url}, found {names}", file=sys.stderr)
            return 1
        partial_path = output_path + ".part"
        with archive.open(names[0]) as member, open(partial_path, "wb") as out:
            shutil.copyfileobj(member, out, 8 << 20)
    # Rename only once complete, so an interrupted download is retried by
    # make instead of being mistaken for a finished one.
    shutil.move(partial_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
