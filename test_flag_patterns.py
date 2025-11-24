import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import mplfinance as mpf
from flag import find_flags_pips
    

data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

data = np.log(data)
dat_slice = data['close'].to_numpy()


orders = list(range(3, 49))
bull_flag_wr = []
bear_flag_wr = []

bull_flag_avg = []
bear_flag_avg = []

bull_flag_count = []
bear_flag_count = []

bull_flag_total_ret = []
bear_flag_total_ret = []


for order in orders:
    bull_flags, bear_flags  = find_flags_pips(dat_slice, order)

    bull_flag_df = pd.DataFrame()
    bear_flag_df = pd.DataFrame()

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

    if len(bull_flag_df) > 0:
        bull_flag_count.append(len(bull_flag_df))
        bull_flag_avg.append(bull_flag_df['return'].mean())
        bull_flag_wr.append(len(bull_flag_df[bull_flag_df['return'] > 0]) / len(bull_flag_df))
        bull_flag_total_ret.append(bull_flag_df['return'].sum())
    else:
        bull_flag_count.append(0)
        bull_flag_avg.append(np.nan)
        bull_flag_wr.append(np.nan)
        bull_flag_total_ret.append(0)
    
    if len(bear_flag_df) > 0:
        bear_flag_count.append(len(bear_flag_df))
        bear_flag_avg.append(bear_flag_df['return'].mean())
        bear_flag_wr.append(len(bear_flag_df[bear_flag_df['return'] > 0]) / len(bear_flag_df))
        bear_flag_total_ret.append(bear_flag_df['return'].sum())
    else:
        bear_flag_count.append(0)
        bear_flag_avg.append(np.nan)
        bear_flag_wr.append(np.nan)
        bear_flag_total_ret.append(0)
    

results_df = pd.DataFrame(index=orders)
results_df['bull_flag_count'] = bull_flag_count
results_df['bull_flag_avg'] = bull_flag_avg
results_df['bull_flag_wr'] = bull_flag_wr
results_df['bull_flag_total'] = bull_flag_total_ret

results_df['bear_flag_count'] = bear_flag_count
results_df['bear_flag_avg'] = bear_flag_avg
results_df['bear_flag_wr'] = bear_flag_wr
results_df['bear_flag_total'] = bear_flag_total_ret

# Plot bull flag results
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Bull Flag Performance", fontsize=16)
results_df['bull_flag_count'].plot.bar(ax=ax[0,0], color='green')
results_df['bull_flag_avg'].plot.bar(ax=ax[0,1], color='red')
results_df['bull_flag_total'].plot.bar(ax=ax[1,0], color='blue')
results_df['bull_flag_wr'].plot.bar(ax=ax[1,1], color='yellow')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found', pad=5)
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return', pad=5)
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns', pad=5)
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate', pad=5)
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')

# Fix x-axis labels for all subplots
for i in range(2):
    for j in range(2):
        ax[i,j].tick_params(axis='x', rotation=45, labelsize=6)

# Adjust subplot spacing to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)  # More space between subplots
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Bear Flag Performance", fontsize=20)
results_df['bear_flag_count'].plot.bar(ax=ax[0,0], color='green')
results_df['bear_flag_avg'].plot.bar(ax=ax[0,1], color='red')
results_df['bear_flag_total'].plot.bar(ax=ax[1,0], color='blue')
results_df['bear_flag_wr'].plot.bar(ax=ax[1,1], color='yellow')
ax[0,1].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,0].hlines(0.0, xmin=-1, xmax=len(orders), color='white')
ax[1,1].hlines(0.5, xmin=-1, xmax=len(orders), color='white')
ax[0,0].set_title('Number of Patterns Found', pad=5)
ax[0,0].set_xlabel('Order Parameter')
ax[0,0].set_ylabel('Number of Patterns')
ax[0,1].set_title('Average Pattern Return', pad=5)
ax[0,1].set_xlabel('Order Parameter')
ax[0,1].set_ylabel('Average Log Return')
ax[1,0].set_title('Sum of Returns', pad=5)
ax[1,0].set_xlabel('Order Parameter')
ax[1,0].set_ylabel('Total Log Return')
ax[1,1].set_title('Win Rate', pad=5)
ax[1,1].set_xlabel('Order Parameter')
ax[1,1].set_ylabel('Win Rate Percentage')

# Fix x-axis labels for all subplots
for i in range(2):
    for j in range(2):
        ax[i,j].tick_params(axis='x', rotation=45, labelsize=6)

# Adjust subplot spacing to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)  # More space between subplots
plt.show()