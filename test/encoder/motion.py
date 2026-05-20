import numpy as np

BLOCK = 16
SEARCH = 8


def block_matching(curr, ref, x, y):
   
    h, w = ref.shape
    best_dx, best_dy = 0, 0
    min_error = float('inf')

    if x + BLOCK > h or y + BLOCK > w:
        return 0, 0

    current_block = curr[x:x + BLOCK, y:y + BLOCK]

    for dx in range(-SEARCH, SEARCH + 1):
        for dy in range(-SEARCH, SEARCH + 1):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + BLOCK > h or ny + BLOCK > w:
                continue
            ref_block = ref[nx:nx + BLOCK, ny:ny + BLOCK]
            error = np.sum(np.abs(current_block.astype(np.int16) - ref_block.astype(np.int16)))
            if error < min_error:
                min_error = error
                best_dx, best_dy = dx, dy

    return best_dx, best_dy
