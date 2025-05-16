
# GDEA HBEA

## Data Processing Logic

#### 1. **Import Required Packages**
#### 2. **Load Raw Data**
#### 3. **Inspect Raw Data**
#### 4. **Clean Data Structure**
   - **Objective**: Remove metadata, duplicates, and standardize column names for consistency.
   - **Steps**:
     - **Remove Metadata and Footer**:
       - Drop the first few rows containing metadata (e.g., frequency, units, IDs, source).
       - Drop the last few rows containing footer information (e.g., data source notes).
     - **Handle Duplicate Columns**:
       - Identify and drop duplicate columns (e.g., repeated turnover or volume columns) by name.
     - **Rename Columns**:
       - Create a mapping dictionary to rename columns to standardized names (e.g., `date`, `close`, `vwap`, `volume_tons`, `turnover_cny`, `cum_turnover_cny`).
     - **Sort Data**:
       - Sort the DataFrame by the `date` column in ascending order to ensure chronological order.
   - **Output**: Cleaned DataFrame (`df`) with standardized columns, no metadata or duplicates, and sorted by date.

#### 5. **Parse Data Types**
   - **Objective**: Convert columns to appropriate data types for analysis and ensure a consistent index.
   - **Steps**:
     - Convert the `date` column to `datetime` format and set it as the DataFrame index.
     - Convert numeric columns (`close`, `vwap`, `volume_tons`, `turnover_cny`, `cum_turnover_cny`) to `float64` to handle decimal values and ensure numerical operations.
     - Verify data types using `df.dtypes`.
   - **Output**: DataFrame with `date` as the index (datetime format) and numeric columns as `float64`.

#### 6. **Align with Trading Calendar**
   - **Objective**: Align the dataset with a trading calendar to distinguish trading and non-trading days.
   - **Steps**:
     - Use `exchange_calendars` to retrieve the trading sessions for the relevant exchange (e.g., XSHG for Shanghai Stock Exchange).
     - Create a complete date range from the earliest to the latest date in the dataset.
     - Reindex the DataFrame to include all calendar days, filling missing dates with `NaN`.
     - Add a boolean column `is_trading_day` to indicate whether each date is a trading day based on the exchange calendar.
   - **Output**: DataFrame (`df_restored`) with all calendar days, indexed by date, and a `is_trading_day` column.

#### 7. **Analyze Trading vs. Non-Trading Days**
   - **Objective**: Assess data availability on trading and non-trading days to inform imputation strategy.
   - **Steps**:
     - Split the DataFrame into trading (`df_trading`) and non-trading (`df_non_trading`) days based on `is_trading_day`.
     - Calculate the number of non-trading days with activity (e.g., non-`NaN` values in `volume_tons`).
     - Calculate the number of trading days with all `NaN` values (indicating no trading activity).
     - Visualize the distribution using pie charts:
       - Left pie chart: Non-trading days (activity vs. quiet).
       - Right pie chart: Trading days (all `NaN` vs. has data).
   - **Output**: Insights into data completeness, with visualizations showing the proportion of active vs. quiet days.

#### 8. **Fill Missing Values**
   - **Objective**: Impute missing values based on trading status and market logic to create a consistent dataset.
   - **Steps**:
     - **Generate Flags**:
       - `is_open`: True if the date is a trading day (`is_trading_day`).
       - `is_quiet`: True if `is_open` and `volume_tons` is `NaN` (indicating no trading activity).
       - `has_trade`: True if `volume_tons` is not `NaN` (indicating trading activity).
       - Verify flag counts using `value_counts()`.
     - **Activity Columns** (`volume_tons`, `turnover_cny`):
       - Set to `0.0` for `is_quiet` days (open market, no trades).
       - Leave as `NaN` for non-trading days (`is_open == False`).
       - Retain original values where `has_trade` is True.
       - Verify no unfilled values remain on open days.
     - **Price Columns** (`close`, `vwap`):
       - Forward-fill missing prices for one trading day only within `is_open` blocks using `groupby` and `ffill(limit=1)`.
       - (Note: The notebooks mention creating `close_carried` and `vwap_carried` flags but do not implement them.)
       - Verify the number of filled and remaining `NaN` values on open days.
       - Check for unintended fills (e.g., non-target days filled).
     - **Cumulative Turnover** (`cum_turnover_cny`):
       - Forward-fill without limit to propagate the last known cumulative value.
       - Verify no unfilled values remain on open days.
   - **Output**: DataFrame with imputed values for activity and price columns, and fully filled cumulative turnover.

#### 9. **Clean Up**
   - **Objective**: Finalize the DataFrame by removing temporary columns and ensuring data consistency.
   - **Steps**:
     - Drop the `is_trading_day` column as it is redundant with `is_open`.
     - Drop rows where `close` and `vwap` are still `NaN` (e.g., before the market started trading).
     - Ensure all numeric columns are `float64` (already handled in parsing).
     - Verify the final DataFrame structure using `head()` or `info()`.
   - **Output**: Final cleaned DataFrame with no redundant columns and consistent data types.