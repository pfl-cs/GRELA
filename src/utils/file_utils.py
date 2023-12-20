import pickle
import os

def load_str_int_map(filepath, delim="||", key_pos=0, val_pos=1):
    m = {}
    with open(filepath, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            terms = line.strip().split(delim)
            key = terms[key_pos]
            value = int(terms[val_pos])
            m[key] = value
    return m

def load_int_str_map(filepath, delim="||", key_pos=0, val_pos=1):
    m = {}
    with open(filepath, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            terms = line.strip().split(delim)
            key = int(terms[key_pos])
            value = terms[val_pos]
            m[key] = value
    return m

def load_str_str_map(filepath, delim="||", key_pos=0, val_pos=1):
    m = {}
    with open(filepath, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            terms = line.strip().split(delim)
            key = terms[key_pos]
            if val_pos < 0:
                value = line
            else:
                value = terms[val_pos]
            m[key] = value
    return m


def load_str_str_map_from_two_part_lines(filepath, two_part_delim="|*|", delim_at_first_part="||", key_pos=0):
    m = {}
    with open(filepath, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            two_parts = line.strip().split(two_part_delim)
            first_part = two_parts[0]
            terms = first_part.split(delim_at_first_part)
            key = terms[key_pos]
            val = two_parts[1]
            m[key] = val
    return m


def read_all_lines(path):
    with open(path, 'r') as reader:
        lines = reader.readlines()
    return lines

def read_first_line(path):
    with open(path, 'r') as reader:
        line = reader.readline()
    return line

def read_all(path):
    with open(path, 'r') as reader:
        s = reader.read()
    return s

def write_all(path, s):
    with open(path, 'w') as writer:
        writer.write(s)


def write_all_lines(path, lines):
    with open(path, 'w') as writer:
        writer.writelines(lines)


def write_pickle(path, data):
    with open(path, "wb") as writer:
        pickle.dump(data, writer)

def read_pickle(path):
    with open(path, "rb") as reader:
        data = pickle.load(reader)
    return data


def cmp_two_files(file1, file2):
    lines1 = read_all_lines(file1)
    lines2 = read_all_lines(file2)
    if len(lines1) != len(lines2):
        print('file1.length = {0:d}, file2.length = {1:d}', len(lines1), len(lines2))
        return False
    for i, line1 in enumerate(lines1):
        line2 = lines2[i]
        if line1 != line2:
            print('At line ', i + 1)
            print('line1 = ', line1)
            print('line2 = ', line2)
            return False
    return True

