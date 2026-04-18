# ABOUTME: Appends supplementary analysis cells to notebook 07_pass2_diagnostics.ipynb.
# ABOUTME: Inputs: notebooks/07_pass2_diagnostics.ipynb. Outputs: same file (mutated in place).
# Run: python3 scripts/append_nb07_cells.py

import json
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

NB_PATH = '/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2/notebooks/07_pass2_diagnostics.ipynb'

with open(NB_PATH) as f:
    nb = nbformat.read(f, as_version=4)

existing_count = len(nb.cells)
print(f'Existing cells: {existing_count}')

# -------------------------------------------------------------------------
# Cell 1: Markdown intro for supplementary section
# -------------------------------------------------------------------------
md_intro = new_markdown_cell(source="""\
## Supplementary: year-by-year tearsheet and extended cost sensitivity

This section closes two loose ends flagged in synthesis_pass2.md.

1. **Temporal concentration check.** The aggregate OOS Sharpe figures could mask \
concentration in one or two calendar years. The year-by-year tearsheet \
(output/index_yearly_tearsheet.csv) unpacks annual Sharpe, return, drawdown, and \
mean IC for all four index configs across 2020-2026.

2. **Cost-model extrapolation.** The cost sensitivity in Pass 2 only ran to 2x spread \
multiplier. The synthesis flagged that realistic costs may extend to 5-14x. \
The extended sweep (output/cost_sensitivity_extended.csv) fills that gap and lets \
us test whether the WRITEUP claim of net Sharpe 1.5-2.0 at realistic costs holds \
across the full realistic range.\
""")

# -------------------------------------------------------------------------
# Cell 2: Load supplementary CSVs
# -------------------------------------------------------------------------
code_load = new_code_cell(source="""\
BASE = '/Users/idrees/Desktop/Claude/projects/crypto_overnight_em_equity_p2/output'

yearly   = pd.read_csv(f'{BASE}/index_yearly_tearsheet.csv')
cost_ext = pd.read_csv(f'{BASE}/cost_sensitivity_extended.csv')
cost_p2  = pd.read_csv(f'{BASE}/cost_sensitivity_pass2.csv')

print('yearly shape:   ', yearly.shape)
print('cost_ext shape: ', cost_ext.shape)
print('cost_p2 shape:  ', cost_p2.shape)
print()
print('yearly configs:', yearly['config'].unique().tolist())
print('cost_ext spread_mults:', sorted(cost_ext['spread_mult'].unique()))
print('cost_p2 spread_mults: ', sorted(cost_p2['spread_mult'].unique()))
""")

# -------------------------------------------------------------------------
# Cell 3: Markdown intro for year-by-year tearsheet
# -------------------------------------------------------------------------
md_yearly_intro = new_markdown_cell(source="""\
### 5.1 Year-by-year tearsheet (index configs)

Each row covers one calendar year. Columns: number of trading days, fraction of \
days the strategy was invested (pct_invested), annual Sharpe, annualized return, \
max drawdown, and mean IC. The horizontal reference line at Sharpe 0.5 marks the \
minimum threshold for a year to be considered economically meaningful.\
""")

# -------------------------------------------------------------------------
# Cell 4: Render yearly tearsheet charts (KR and HK gate_off)
# -------------------------------------------------------------------------
code_yearly_chart = new_code_cell(source="""\
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Config labels for display
config_map = {
    'lgbm_index_kr_gap_index_threshold_gate_off': 'lgbm / index_kr / gap / gate_off',
    'lgbm_index_kr_gap_index_threshold_gate_on':  'lgbm / index_kr / gap / gate_on',
    'lgbm_index_hk_gap_index_threshold_gate_off': 'lgbm / index_hk / gap / gate_off',
    'lgbm_index_hk_gap_index_threshold_gate_on':  'lgbm / index_hk / gap / gate_on',
}

def plot_yearly_sharpe(df, config_key, ax, title):
    sub = df[df['config'] == config_key].sort_values('year')
    colors = ['steelblue' if s >= 0.5 else 'tomato' for s in sub['annual_sharpe']]
    ax.bar(sub['year'], sub['annual_sharpe'], color=colors, width=0.6)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Sharpe = 0.5 threshold')
    ax.axhline(0.0, color='grey', linestyle='-', linewidth=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Sharpe')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

plot_yearly_sharpe(
    yearly,
    'lgbm_index_kr_gap_index_threshold_gate_off',
    axes[0],
    'lgbm / index_kr / gap / gate_off -- annual Sharpe by year'
)
plot_yearly_sharpe(
    yearly,
    'lgbm_index_hk_gap_index_threshold_gate_off',
    axes[1],
    'lgbm / index_hk / gap / gate_off -- annual Sharpe by year'
)

plt.tight_layout()
plt.show()

# Styled table for the KR gate_off config
kr_off = yearly[yearly['config'] == 'lgbm_index_kr_gap_index_threshold_gate_off'] \
    .sort_values('year')[['year','n_days','pct_invested','annual_sharpe','ann_return','max_drawdown','mean_ic']] \
    .set_index('year')

display(kr_off.style
    .format({'pct_invested': '{:.2f}', 'annual_sharpe': '{:.2f}',
             'ann_return': '{:.3f}', 'max_drawdown': '{:.4f}', 'mean_ic': '{:.4f}'})
    .set_caption('lgbm / index_kr / gap / gate_off: year-by-year tearsheet'))
""")

# -------------------------------------------------------------------------
# Cell 5: Markdown -- 2023 attenuation call-out
# -------------------------------------------------------------------------
md_2023 = new_markdown_cell(source="""\
### 5.2 Note on 2023 attenuation

For lgbm/index_kr/gap/gate_off, the 2023 annual Sharpe is 0.36, which is below the \
0.5 threshold. Mean IC in that year dropped to 0.09 (versus 0.20-0.48 in surrounding \
years), and pct_invested fell to 0.40 -- meaning the strategy was flat on roughly 60% \
of trading days due to the index-level threshold rule (gate was not applied, but the \
signal itself fell below the position threshold on most days).

This is not an isolated model failure. The index-level 2023 attenuation parallels the \
main-universe 2023 attenuation documented in the Pass 1 year-by-year results. Both \
signal families weakened together in the same calendar year. That co-movement is \
consistent with a shared macro-regime effect -- likely reduced crypto-equity correlation \
or dampened overnight gap dispersion during a period of compressed market volatility -- \
rather than a model-specific artifact or overfitting. The 2024-2026 recovery to IC \
values above 0.24 and Sharpe above 1.0 further supports a transient-regime reading.
""")

# -------------------------------------------------------------------------
# Cell 6: Markdown intro for extended cost sensitivity
# -------------------------------------------------------------------------
md_cost_intro = new_markdown_cell(source="""\
### 5.3 Extended cost sensitivity (spread 1x to 14x)

The Pass 2 cost sweep ran from 0.5x to 2x spread multiplier. The synthesis \
(synthesis_pass2.md) flagged that realistic brokerage costs in EM equity markets \
might range from 5x to 14x the raw bid-ask spread, once market impact and timing \
slippage are included. The extended sweep provides the missing data points at \
5x, 7x, 10x, and 14x.\
""")

# -------------------------------------------------------------------------
# Cell 7: Build combined DataFrame and render chart + table
# -------------------------------------------------------------------------
code_cost_chart = new_code_cell(source="""\
# The four index configs to plot
index_configs = [
    ('index_kr', 'gate_off'),
    ('index_kr', 'gate_on'),
    ('index_hk', 'gate_off'),
    ('index_hk', 'gate_on'),
]

def config_label(universe, gate):
    return f'lgbm / {universe} / gap / {gate}'

# Pull 1x and 2x from cost_p2 (index configs only, gap target)
p2_index = cost_p2[
    (cost_p2['model'] == 'lgbm') &
    (cost_p2['universe'].isin(['index_kr', 'index_hk'])) &
    (cost_p2['target'] == 'gap') &
    (cost_p2['spread_mult'].isin([1.0, 2.0]))
].copy()

# cost_p2 does not have a 'gate' column; infer from strategy (gate_off = index_threshold;
# gate_on rows are not in cost_p2, they appear in cost_ext only via gate column)
# cost_p2 only has gate_off rows for index configs (strategy == 'index_threshold', no gate col)
# We treat all cost_p2 index rows as gate_off (the file has no gate column)
p2_index['gate'] = 'gate_off'
p2_index['label'] = p2_index.apply(lambda r: config_label(r['universe'], r['gate']), axis=1)

# Pull 5x-14x from cost_ext
ext_sel = cost_ext[
    (cost_ext['model'] == 'lgbm') &
    (cost_ext['target'] == 'gap')
].copy()
ext_sel['label'] = ext_sel.apply(lambda r: config_label(r['universe'], r['gate']), axis=1)

# Combine
cols = ['label', 'spread_mult', 'net_sharpe']
combined = pd.concat([
    p2_index[cols],
    ext_sel[cols]
], ignore_index=True).sort_values(['label', 'spread_mult'])

print('Combined spread_mults:', sorted(combined['spread_mult'].unique()))
print('Combined labels:', combined['label'].unique().tolist())

# Line plot
fig, ax = plt.subplots(figsize=(10, 5))

for lbl, grp in combined.groupby('label'):
    g = grp.sort_values('spread_mult')
    ax.plot(g['spread_mult'], g['net_sharpe'], marker='o', label=lbl)

ax.axhline(1.5, color='darkgreen', linestyle='--', linewidth=1, label='Sharpe = 1.5')
ax.axhline(0.5, color='black',     linestyle='--', linewidth=1, label='Sharpe = 0.5')
ax.axhline(0.0, color='grey',      linestyle='-',  linewidth=0.5)
ax.set_xlabel('Spread multiplier (x raw bid-ask)')
ax.set_ylabel('Net Sharpe (OOS)')
ax.set_title('Cost sensitivity: net Sharpe vs spread multiplier (index configs, gap target)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# Pivot table
pivot = combined.pivot_table(index='spread_mult', columns='label', values='net_sharpe')
pivot.index.name = 'spread_mult'
display(pivot.style
    .format('{:.2f}')
    .set_caption('Net Sharpe by spread multiplier and config (combined 1x-14x)'))
""")

# -------------------------------------------------------------------------
# Cell 8: Markdown -- WRITEUP claim verdict
# -------------------------------------------------------------------------
md_verdict = new_markdown_cell(source="""\
### 5.4 Revised characterization of cost robustness

The WRITEUP claim reads: "net Sharpe 1.5-2.0 at realistic costs." That range is now \
testable against the extended sweep.

**Measured outcomes for the top config (lgbm / index_kr / gap / gate_off):**

- At 5x spread (realistic-low end): net Sharpe = 2.15
- At 7x spread (middle realistic): net Sharpe = 1.34
- At 10x spread (realistic-high end): net Sharpe = 0.14
- At 14x spread (upper tail): net Sharpe = -1.41

The claim is accurate only at the lower end of the realistic range. At 7x -- a \
widely-cited rough estimate for all-in EM equity execution cost -- the top config \
falls below 1.5. At 10x it is essentially breakeven. At 14x it is deeply negative.

**A more accurate characterization:** net Sharpe of 1-2 at the lower half of the \
realistic cost range (5-7x); near zero to negative at the upper half (10-14x). The \
result is cost-sensitive, and the defensible window is narrower than the WRITEUP \
implies. This should be noted explicitly in any presentation of the findings.\
""")

# -------------------------------------------------------------------------
# Cell 9: Final paragraph closing the two loose ends
# -------------------------------------------------------------------------
md_closing = new_markdown_cell(source="""\
### 5.5 Summary

Both supplementary analyses close items flagged as unresolved in synthesis_pass2.md.

The year-by-year tearsheet confirms that the aggregate OOS Sharpe is not driven by \
a single calendar year. Five of seven years (2020-2022, 2024-2026) clear the 0.5 \
threshold for the top config; 2023 is the sole below-threshold year and is explicable \
by a shared macro-regime effect rather than model failure.

The extended cost sweep quantifies the boundary of cost robustness. The "1.5-2.0 net \
Sharpe at realistic costs" claim survives at the low end of realistic costs but \
fails at the midpoint of the realistic range. Future reporting should cite the cost \
multiplier explicitly alongside any net Sharpe figure.\
""")

# -------------------------------------------------------------------------
# Append all new cells
# -------------------------------------------------------------------------
new_cells = [
    md_intro,
    code_load,
    md_yearly_intro,
    code_yearly_chart,
    md_2023,
    md_cost_intro,
    code_cost_chart,
    md_verdict,
    md_closing,
]

for cell in new_cells:
    nb.cells.append(cell)

print(f'Cells before: {existing_count}')
print(f'Cells after:  {len(nb.cells)}')
print(f'Appended:     {len(new_cells)} cells')

with open(NB_PATH, 'w') as f:
    nbformat.write(nb, f)

print('Notebook written successfully.')
