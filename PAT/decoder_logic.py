import numpy as np

def get_decoder_target(op3, op2, op1, op0, cflag, L, H):
    """
    Returns the expected 6 outputs [SelB, SelA, L0, L1, L2, L3]
    L = Low Voltage Target, H = High Voltage Target
    """
    # Default to ALL LOW
    out = [L, L, L, L, L, L] 
    
    # 0000: ADD A, Im
    if   (op3==0 and op2==0 and op1==0 and op0==0): out = [L, L, H, L, L, L]
    # 0001: MOV A, B
    elif (op3==0 and op2==0 and op1==0 and op0==1): out = [L, H, H, L, L, L]
    # 0010: IN A
    elif (op3==0 and op2==0 and op1==1 and op0==0): out = [L, H, L, H, L, L]
    # 0011: MOV A, Im
    elif (op3==0 and op2==0 and op1==1 and op0==1): out = [L, H, H, H, L, L]
    # 0100: MOV B, A
    elif (op3==0 and op2==1 and op1==0 and op0==0): out = [H, L, L, L, H, L]
    # 0101: ADD B, Im
    elif (op3==0 and op2==1 and op1==0 and op0==1): out = [L, H, L, L, H, H]
    # 0110: IN B
    elif (op3==0 and op2==1 and op1==1 and op0==0): out = [H, L, L, L, H, H]
    # 0111: MOV B, Im
    elif (op3==0 and op2==1 and op1==1 and op0==1): out = [H, L, L, H, H, L]
    # 1001: OUT B
    elif (op3==1 and op2==0 and op1==0 and op0==1): out = [H, L, L, L, H, L]
    # 1011: OUT Im
    elif (op3==1 and op2==0 and op1==1 and op0==1): out = [L, H, L, L, H, L]
    # 1110: JNC
    elif (op3==1 and op2==1 and op1==1 and op0==0):
        # Only if Carry Flag is 0 do we trigger (Wait, your C++ says LOW for everything?)
        # Re-reading your code: JNC (C=0) sets all LOW. JNC (C=1) sets all LOW.
        # It seems JNC purely affects Program Counter logic not handled by these pins?
        # I will output ALL LOW as per your 'else' block
        out = [L, L, L, L, L, L]
    # 1111: JMP
    elif (op3==1 and op2==1 and op1==1 and op0==1): out = [L, L, L, L, L, L]
    
    return out

def generate_full_truth_table(L_targets, H_targets):
    """
    Generates inputs (32 combinations) and targets (32 x 6)
    L_targets: List of 6 Low values (one per channel)
    H_targets: List of 6 High values (one per channel)
    """
    inputs = []
    targets = []
    
    # Iterate 0 to 31 (5 bits)
    for i in range(32):
        # Extract bits
        op3   = (i >> 4) & 1
        op2   = (i >> 3) & 1
        op1   = (i >> 2) & 1
        op0   = (i >> 1) & 1
        cflag = (i >> 0) & 1
        
        inputs.append([op3, op2, op1, op0, cflag])
        
        # We pass the specific L and H for the channels, but for simplicity here
        # we assume scalar L/H. In reality, we need per-channel logic.
        # Let's assume we pass the arrays to the optimizer.
        tgt_row = get_decoder_target(op3, op2, op1, op0, cflag, 0, 1) # Get logic 0/1 first
        
        # Map 0->L_target[ch], 1->H_target[ch]
        phys_row = []
        for ch in range(6):
            if tgt_row[ch] == 0:
                phys_row.append(L_targets[ch])
            else:
                phys_row.append(H_targets[ch])
        
        targets.append(phys_row)

    return np.array(inputs), np.array(targets)