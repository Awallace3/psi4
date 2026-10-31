"""Bounded ELF64/x86-64 dynamic-loader metadata from an already sealed FD.

No subprocess, source pathname, ELF execution, or mutable alias lookup.
Only DT_NEEDED/DT_SONAME are returned; unsupported/malformed images reject.
"""
import os
import struct


def loader_metadata(fd, size):
    def read(offset, count):
        if offset < 0 or count < 0 or count > 64*1024**2 or offset+count > size:
            raise ValueError("ELF metadata range is outside the sealed image")
        result = bytearray()
        while len(result) < count:
            block = os.pread(fd, count-len(result), offset+len(result))
            if not block:
                raise ValueError("Truncated sealed ELF metadata")
            result.extend(block)
        return bytes(result)

    header = struct.unpack("<16sHHIQQQIHHHHHH", read(0,64))
    ident, kind, machine, version = header[:4]
    phoff, ehsize, phsize, phnum = header[5], header[8], header[9], header[10]
    if (ident[:7] != b"\x7fELF\x02\x01\x01" or ident[7] not in (0,3)
            or kind != 3 or machine != 62 or version != 1
            or ehsize != 64 or phsize != 56 or not 1 <= phnum <= 4096):
        raise ValueError("Only bounded Linux x86-64 ELF64 shared images are supported")
    segments = struct.iter_unpack("<IIQQQQQQ", read(phoff,phsize*phnum))
    loads = []
    dynamic = []
    for segment in segments:
        tag, _, offset, address, _, filesz, _, _ = segment
        if offset+filesz > size:
            raise ValueError("ELF program segment exceeds the sealed image")
        if tag == 1:
            loads.append((offset,address,filesz))
        elif tag == 2:
            dynamic.append((offset,filesz))
    if len(dynamic) != 1:
        raise ValueError("Exactly one bounded ELF dynamic segment is required")
    offset,count = dynamic[0]
    if not 16 <= count <= 1024**2 or count % 16:
        raise ValueError("Invalid ELF dynamic segment size")
    needed = []
    table = {}
    terminated = False
    for tag,value in struct.iter_unpack("<qQ", read(offset,count)):
        if tag == 0:
            terminated = True
            break
        if tag in (0x6ffffefa,0x6ffffefb,0x6ffffefc,0x7ffffffd,0x7fffffff):
            raise ValueError("ELF audit/config/filter loader extensions are unsupported")
        if tag == 1:
            needed.append(value)
        elif tag in (5,10,14,15,29):
            if tag in table:
                raise ValueError("Duplicate ELF string-table/SONAME tag")
            table[tag] = value
    if not terminated or 5 not in table or 10 not in table or not table[10]:
        raise ValueError("Incomplete ELF dynamic string table")
    candidates = [position+table[5]-address for position,address,length in loads
                  if address <= table[5] and table[5]+table[10] <= address+length]
    if len(candidates) != 1:
        raise ValueError("Ambiguous/non-file-backed ELF dynamic string table")
    strings = read(candidates[0],table[10])

    def name(index, *, search=False):
        if not 0 <= index < len(strings):
            raise ValueError("ELF dependency string is outside the table")
        end = strings.find(b"\0",index)
        if end < 0:
            raise ValueError("Unterminated ELF dependency string")
        value = strings[index:end].decode("ascii")
        if not value or "/" in value or value in (".","..") or not search and "$" in value:
            raise ValueError("ELF dependencies must use explicit bare SONAMEs")
        return value

    search = tuple(name(table[tag],search=True) for tag in (15,29) if tag in table)
    if any(value not in ("$ORIGIN","${ORIGIN}") for value in search):
        raise ValueError("ELF search paths outside descriptor-local ORIGIN are unsupported")
    return {"needed":tuple(name(index) for index in needed),
            "soname":name(table[14]) if 14 in table else None}
