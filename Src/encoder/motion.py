import numpy as np

BLOCK = 16
SEARCH = 8  
# motion.py - Ajouter une condition pour réduire le temps de calcul
def block_matching(curr, ref, x, y):
    h, w = ref.shape
    best_dx, best_dy = 0, 0
    min_error = float('inf')
    
    # Vérifier si le bloc courant est valide
    if x+BLOCK > h or y+BLOCK > w:
        return 0, 0
    
    current_block = curr[x:x+BLOCK, y:y+BLOCK]
    
    # Recherche exhaustive (SAD au lieu de SSD pour être plus rapide)
    for dx in range(-SEARCH, SEARCH+1):
        for dy in range(-SEARCH, SEARCH+1):
            nx = x + dx
            ny = y + dy
            
            if nx < 0 or ny < 0 or nx+BLOCK > h or ny+BLOCK > w:
                continue
            
            ref_block = ref[nx:nx+BLOCK, ny:ny+BLOCK]
            
            # SAD (Sum of Absolute Differences) - plus rapide
            error = np.sum(np.abs(current_block.astype(np.int16) - ref_block.astype(np.int16)))
            
            if error < min_error:
                min_error = error
                best_dx, best_dy = dx, dy
    
    return best_dx, best_dy