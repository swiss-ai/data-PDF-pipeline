import pandas as pd

N_DIGITS = 0


def get_text_from_blocks(blocks):
    return ''.join([line[4] for line in blocks])


def blocks_by_id(blocks, ids):
    filtered_blocks = [[b[i] for i in ids] for b in blocks]
    return filtered_blocks


def blocks_to_text(blocks, ids):
    blocks_rounded = get_blocks_rounded(blocks)
    filtered_blocks = blocks_by_id(blocks_rounded, ids)
    return '\n'.join(['(' + ','.join(map(str, block)) + ')' for block in filtered_blocks])


def round_coord(number):
    if N_DIGITS == 0:
        return int(number)
    else:
        return round(number, N_DIGITS)


def get_blocks_rounded(blocks):
    return [[round_coord(b[0]), round_coord(b[1]), round_coord(b[2]), round_coord(b[3])] + list(b[4:]) for b in blocks]


def blocks_to_df(blocks):
    df = pd.DataFrame(get_blocks_rounded(blocks), columns=[
        'x0', 'y0', 'x1', 'y1', 'lines in block', 'block_no', 'block_type'])
    df['index'] = df['x0'].astype(
        str) + '-' + df['y0'].astype(str) + '-' + df['x1'].astype(str) + '-' + df['y1'].astype(str)
    df.set_index('index', drop=True, inplace=True)
    df.sort_values('block_no', inplace=True)
    return df


def prompt_lines(blocks, ids):
    blocks_content = blocks_to_text(blocks, ids)
    return "---\nWe have detected following lines:\n---\n\n" + blocks_content + \
           "\n\nPlease comment and fix layout (if there are any mistakes) \n\n"


def get_result_index(res):
    """coords is a list of lists in the structure"""
    return ['-'.join(map(str, coords)) for coords in res]
