# Data Sources Overview

## Carbon Markets Data

### Guangdong (GDEA)
- **Source**: Guangzhou Carbon Emission Exchange
- **Period**: 2014-2024
- **Frequency**: Daily trading data
- **Key Fields**: Open, High, Low, Close, Volume, Amount
- **Coverage**: 2,436 trading days

### Hubei (HBEA)
- **Source**: Hubei Carbon Emission Exchange
- **Period**: 2014-2024  
- **Frequency**: Daily trading data
- **Key Fields**: Open, High, Low, Close, Volume, Amount
- **Coverage**: 2,425 trading days

## Macroeconomic Indicators (21 Features)

### National Level (中国)
1. **CPI** - Consumer Price Index (Monthly)
2. **GDP** - Current and constant price (Quarterly)
3. **Manufacturing PMI** - Purchasing Managers Index (Monthly)
4. **Industrial Production**:
   - Crude Oil Processing (Monthly)
   - Coal Production (Monthly)
   - Cement Production (Monthly)
   - Crude Steel Production (Monthly)
5. **Electricity**:
   - Total Electricity Consumption (Monthly)
   - Thermal Power Generation (Monthly)
6. **Financial**: Social Financing Scale (Monthly)
7. **Exchange Rate**: USD/CNY spot rate (Daily)

### Regional Level
8. **Guangdong Province**:
   - GDP (Quarterly)
   - Industrial Value Added (Monthly)
   - Electricity Consumption (Monthly)

9. **Hubei Province**:
   - GDP (Quarterly)
   - Industrial Value Added (Monthly)
   - Electricity Consumption (Monthly)

### International Markets
10. **Energy Prices**:
    - Brent Crude Oil Futures (Daily)
    - NYMEX Natural Gas (Daily)
    - European ARA Coal Spot (Daily)
11. **Carbon Market**: EU ETS Futures (Daily)

## Policy Documents

### Sources and Coverage
- **MEE (生态环境部)**: 573 documents
  - Decrees: 89
  - Notices: 484
  
- **HBETS (湖北碳交所)**: 684 documents
  - Center Dynamics: 684
  
- **GZETS (广州碳交所)**: 2,055 documents
  - Trading Announcements: 684
  - Center News: 876
  - Provincial/Municipal: 495

### Filtering Results
- **Total Scraped**: 3,312 documents
- **Carbon-Relevant**: 2,617 documents (79% retention)
- **Time Span**: 2015-2025

## Data Quality Notes

### Trading Calendar
All data aligned to Shanghai Stock Exchange (XSHG) calendar for consistency.

### Missing Value Treatment
1. Forward-fill for continuity
2. Linear interpolation for smoothing
3. Daily alignment to trading calendar

### Lag Structure
- **Daily data**: 1-day lag (T-1)
- **Monthly/Quarterly data**: 15-day lag (T-15)
  
This respects real-world data publication delays.