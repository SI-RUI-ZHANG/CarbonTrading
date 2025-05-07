# GDEA Preprocessing Outline

1. **Load raw CSV**  
   - Skip rows **0–3** (Wind metadata) and the last **2** footer lines (“数据来源：Wind”).

2. **Assign & inspect header columns**  
   After loading, confirm you have 7 columns. Their original names and integer indices are:  
   | Index | 原始列名                                               |
   |-------|--------------------------------------------------------|
   | 0     | `指标名称` (日期)                                       |
   | 1     | `广东:收盘价:碳排放权配额(GDEA)`                         |
   | 2     | `广东:成交均价:碳排放权配额(GDEA)`                       |
   | 3     | `广东:当日成交量:碳排放权配额(GDEA)`                     |
   | 4     | `广东:当日成交额:碳排放权配额(GDEA)`                     |
   | 5     | `广东:当日成交额:碳排放权配额(GDEA).1` (重复)            |
   | 6     | `广东:累计成交额:碳排放权配额(GDEA)`                     |

3. **Drop duplicate & unused columns**  
   - Remove column **5** (duplicate of column 4).

4. **Rename columns to code-friendly names**  
   | 原索引 | 原始列名                                             | New name           |
   |--------|------------------------------------------------------|--------------------|
   | 0      | `指标名称`                                           | `date`             |
   | 1      | `广东:收盘价:碳排放权配额(GDEA)`                     | `close`            |
   | 2      | `广东:成交均价:碳排放权配额(GDEA)`                   | `vwap`             |
   | 3      | `广东:当日成交量:碳排放权配额(GDEA)`                 | `volume_tons`      |
   | 4      | `广东:当日成交额:碳排放权配额(GDEA)`                 | `turnover_cny`     |
   | 6      | `广东:累计成交额:碳排放权配额(GDEA)`                 | `cum_turnover_cny` |

5. **Parse & set datetime index**  
   - Convert column `date` to `datetime64`  
   - Set `date` as DataFrame index  
   - Sort index ascending  
   - Confirm **4 157** unique dates; drop any duplicates.

6. **Handle missing values**  
   - `close`, `vwap` → forward-fill; drop leading NaNs if any  
   - `volume_tons`, `turnover_cny` → fill NaN with **0**  
   - `cum_turnover_cny` → forward-fill; verify non-decreasing

7. **Data integrity checks**  
   - Flag any `close` ≤ 0 → set to NaN → forward-fill  
   - Check `turnover_cny ≈ close × volume_tons` within tolerance  
   - Identify one-day jumps > ±50 % in `close` → manual review

8. **Derive additional fields**  
   - `pct_change` = `close`.pct_change()  
   - `log_ret` = log(`close` / `close`.shift(1))  
   - `rolling_vol_20d` = `log_ret`.rolling(20).std()

9. **Align to master trading calendar**  
   - Generate calendar from cleaned index  
   - Reindex if needed when merging with other assets; forward-fill prices

10. **Export cleaned dataset**  
    - Save as Parquet: `GDEA_cleaned_YYYYMMDD.parquet`  
    - Embed git commit or script hash in metadata for reproducibility