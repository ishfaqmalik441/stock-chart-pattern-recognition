import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from pips import find_pips
from rolling_window import rw_top, rw_bottom
from dataclasses import dataclass

@dataclass
class FlagPattern:
    base_x: int         # Start of the trend index, base of pole
    base_y: float       # Start of trend price

    tip_x: int   = -1       # Tip of pole, start of flag
    tip_y: float = -1.

    conf_x: int   = -1      # Index where pattern is confirmed
    conf_y: float = -1.      # Price where pattern is confirmed

    pennant: bool = False      # True if pennant, false if flag

    flag_width: int    = -1
    flag_height: float = -1.

    pole_width: int    = -1
    pole_height: float = -1.

    # Upper and lower lines for flag, intercept is tip_x
    support_intercept: float = -1.
    support_slope: float = -1.
    resist_intercept: float = -1.
    resist_slope: float = -1.

def check_bear_pattern_pips(pending: FlagPattern, data: np.array, i:int, order:int):
    
    # Find max price since local bottom, (top of pole)
    data_slice = data[pending.base_x: i + 1] # i + 1 includes current price
    min_i = data_slice.argmin() + pending.base_x # Min index since local top
    
    if i - min_i < max(5, order * 0.5): # Far enough from max to draw potential flag/pennant
        return False
    
    # Test flag width / height 
    pole_width = min_i - pending.base_x
    flag_width = i - min_i
    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    pole_height = pending.base_y - data[min_i] 
    flag_height = data[min_i:i+1].max() - data[min_i] 
    if flag_height > pole_height * 0.5: # Flag should smaller vertically than preceding trend
        return False

    # If here width/height are OK.
    
    # Find perceptually important points from pole to current time
    pips_x, pips_y = find_pips(data[min_i:i+1], 5, 3) # Finds pips between max and current index (inclusive)

    # Check center pip is less than two adjacent. /\/\ 
    if not (pips_y[2] < pips_y[1] and pips_y[2] < pips_y[3]):
        return False
    
    # Find slope and intercept of flag lines
    # intercept is at the max value (top of pole)
    support_rise = pips_y[2] - pips_y[0]
    support_run = pips_x[2] - pips_x[0]
    support_slope = support_rise / support_run
    support_intercept = pips_y[0] 

    resist_rise = pips_y[3] - pips_y[1]
    resist_run = pips_x[3] - pips_x[1]
    resist_slope = resist_rise / resist_run
    resist_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * resist_slope

    # Find x where two lines intersect.
    #print(pips_x[0], resist_slope, support_slope)
    if resist_slope != support_slope: # Not parallel
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
        #print("Intersects at", intersection)
    else:
        intersection = -flag_width * 100

    # No intersection in flag area
    if intersection <= pips_x[4] and intersection >= 0:
        return False

    # Check if current point has a breakout of flag. (confirmation)
    support_endpoint = pips_y[0] + support_slope * pips_x[4]
    if pips_y[4] > support_endpoint:
        return False
    
    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False
    
    # Filter harshly diverging lines
    if intersection < 0 and intersection > -flag_width:
        return False

    pending.tip_x = min_i
    pending.tip_y = data[min_i]
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept
    

    return True

def check_bull_pattern_pips(pending: FlagPattern, data: np.array, i:int, order:int):
    
    # Find max price since local bottom, (top of pole)
    data_slice = data[pending.base_x: i + 1] # i + 1 includes current price
    max_i = data_slice.argmax() + pending.base_x # Max index since bottom
    pole_width = max_i - pending.base_x
    
    if i - max_i < max(5, order * 0.5): # Far enough from max to draw potential flag/pennant
        return False

    flag_width = i - max_i
    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    pole_height = data[max_i] - pending.base_y 
    flag_height = data[max_i] - data[max_i:i+1].min()
    if flag_height > pole_height * 0.5: # Flag should smaller vertically than preceding trend
        return False

    pips_x, pips_y = find_pips(data[max_i:i+1], 5, 3) # Finds pips between max and current index (inclusive)

    # Check center pip is greater than two adjacent. \/\/  
    if not (pips_y[2] > pips_y[1] and pips_y[2] > pips_y[3]):
        return False

    # Find slope and intercept of flag lines
    # intercept is at the max value (top of pole)
    resist_rise = pips_y[2] - pips_y[0]
    resist_run = pips_x[2] - pips_x[0]
    resist_slope = resist_rise / resist_run
    resist_intercept = pips_y[0] 

    support_rise = pips_y[3] - pips_y[1]
    support_run = pips_x[3] - pips_x[1]
    support_slope = support_rise / support_run
    support_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * support_slope

    # Find x where two lines intersect.
    if resist_slope != support_slope: # Not parallel
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
    else:
        intersection = -flag_width * 100

    # No intersection in flag area
    if intersection <= pips_x[4] and intersection >= 0:
        return False
    
    # Filter harshly diverging lines
    if intersection < 0 and intersection > -1.0 * flag_width:
        return False

    # Check if current point has a breakout of flag. (confirmation)
    resist_endpoint = pips_y[0] + resist_slope * pips_x[4]
    if pips_y[4] < resist_endpoint:
        return False

    # # Pattern is confiremd, fill out pattern details in pending
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    pending.tip_x = max_i
    pending.tip_y = data[max_i]
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept
    
    return True

def find_flags_pips(data: np.array, order:int):
    assert(order >= 3)
    pending_bull = None # Pending pattern
    pending_bear = None # Pending pattern

    bull_flags = []
    bear_flags = []
    for i in range(len(data)):

        # Pattern data is organized like so:
        if rw_top(data, i, order):
            pending_bear = FlagPattern(i - order, data[i - order])
        
        if rw_bottom(data, i, order):
            pending_bull = FlagPattern(i - order, data[i - order])

        if pending_bear is not None:
            if check_bear_pattern_pips(pending_bear, data, i, order):
                if (not pending_bear.pennant):
                    bear_flags.append(pending_bear)
                pending_bear = None

        if pending_bull is not None:
            if check_bull_pattern_pips(pending_bull, data, i, order):
                if (not pending_bull.pennant):
                    bull_flags.append(pending_bull)
                pending_bull = None

    return bull_flags, bear_flags


def plot_flag(candle_data: pd.DataFrame, pattern: FlagPattern, pad=2):
    if pad < 0:
        pad = 0

    start_i = pattern.base_x - pad
    end_i = pattern.conf_x + 1 + pad
    dat = candle_data.iloc[start_i:end_i]
    idx = dat.index
    
    plt.style.use('dark_background')
    fig = plt.gcf()
    ax = fig.gca()

    tip_idx = idx[pattern.tip_x - start_i]
    conf_idx = idx[pattern.conf_x - start_i]

    pole_line = [(idx[pattern.base_x - start_i], pattern.base_y), (tip_idx, pattern.tip_y)]
    upper_line = [(tip_idx, pattern.resist_intercept), (conf_idx, pattern.resist_intercept + pattern.resist_slope * pattern.flag_width)]
    lower_line = [(tip_idx, pattern.support_intercept), (conf_idx, pattern.support_intercept + pattern.support_slope * pattern.flag_width)]

    mpf.plot(dat, alines=dict(alines=[pole_line, upper_line, lower_line], colors=['w', 'b', 'b']), type='line', style='charles', ax=ax)
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    data = np.log(data)

    dat_slice = data['close'].to_numpy()
    bull_flags, bear_flags  = find_flags_pips(dat_slice, 20)

    bull_flag_df = pd.DataFrame()
    bear_flag_df = pd.DataFrame()
    plot_flag(data, bull_flags[3]);
    plot_flag(data, bear_flags[10]);

    # Assemble data into dataframe
    hold_mult = 1.0 # Multipler of flag width to hold for after a pattern
    for i, flag in enumerate(bull_flags):
        bull_flag_df.loc[i, 'flag_width'] = flag.flag_width
        bull_flag_df.loc[i, 'flag_height'] = flag.flag_height
        bull_flag_df.loc[i, 'pole_width'] = flag.pole_width
        bull_flag_df.loc[i, 'pole_height'] = flag.pole_height
        bull_flag_df.loc[i, 'slope'] = flag.resist_slope

        hp = int(flag.flag_width * hold_mult)
        if flag.conf_x + hp >= len(data):
            bull_flag_df.loc[i, 'return'] = np.nan
        else:
            ret = dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x]
            bull_flag_df.loc[i, 'return'] = ret 

    for i, flag in enumerate(bear_flags):
        bear_flag_df.loc[i, 'flag_width'] = flag.flag_width
        bear_flag_df.loc[i, 'flag_height'] = flag.flag_height
        bear_flag_df.loc[i, 'pole_width'] = flag.pole_width
        bear_flag_df.loc[i, 'pole_height'] = flag.pole_height
        bear_flag_df.loc[i, 'slope'] = flag.support_slope

        hp = int(flag.flag_width * hold_mult)
        if flag.conf_x + hp >= len(data):
            bear_flag_df.loc[i, 'return'] = np.nan
        else:
            ret = -1 * (dat_slice[flag.conf_x + hp] - dat_slice[flag.conf_x])
            bear_flag_df.loc[i, 'return'] = ret 