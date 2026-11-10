"""Strict, immutable UTF-8 prefix operations used by the byte tokenizer trie."""


def advance_utf8(prefix: bytes, byte: int):
    """Return (remaining prefix, scalar, reachable range), or None if invalid.

    A range is returned only for an incomplete scalar. The bounds exclude UTF-8
    overlong forms, surrogate code points and values above U+10FFFF.
    """
    data = prefix + bytes((byte,))
    first = data[0]
    if first < 0x80:
        return (b'', chr(first), None) if len(data) == 1 else None
    if 0xc2 <= first <= 0xdf:
        length = 2
    elif 0xe0 <= first <= 0xef:
        length = 3
    elif 0xf0 <= first <= 0xf4:
        length = 4
    else:
        return None
    if len(data) > length:
        return None
    for i, b in enumerate(data[1:], 1):
        low, high = 0x80, 0xbf
        if i == 1:
            if first == 0xe0: low = 0xa0
            if first == 0xed: high = 0x9f
            if first == 0xf0: low = 0x90
            if first == 0xf4: high = 0x8f
        if not low <= b <= high:
            return None
    if len(data) == length:
        return b'', data.decode('utf-8'), None
    low_bytes = bytearray(data)
    high_bytes = bytearray(data)
    while len(low_bytes) < length:
        i = len(low_bytes)
        low = 0xa0 if i == 1 and first == 0xe0 else 0x90 if i == 1 and first == 0xf0 else 0x80
        high = 0x9f if i == 1 and first == 0xed else 0x8f if i == 1 and first == 0xf4 else 0xbf
        low_bytes.append(low)
        high_bytes.append(high)
    return data, None, (ord(low_bytes.decode('utf-8')), ord(high_bytes.decode('utf-8')))
