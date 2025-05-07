# Carbon Trading Project

## Project Structure

```
00_Project_Management/
├── Proposals/              # Project proposals
├── Timelines_Tasks/        # Project timelines and tasks
└── Meeting_Notes/          # Meeting records

01_Data_Raw/
├── Carbon_Markets/
│   ├── GDEA/              # Guangdong ETS data
│   ├── HBEA/              # Hubei ETS data
│   ├── CEA/               # China ETS data
│   └── EU_ETS/
│       ├── EEX/           # European Energy Exchange data
│       ├── ICE/           # Intercontinental Exchange data
│       └── EEA/           # European Environment Agency data
├── Macroeconomic_Data/
│   ├── China/
│   │   ├── NBS/          # National Bureau of Statistics data
│   │   ├── PBOC/         # People's Bank of China data
│   │   └── NEA/          # National Energy Administration data
│   └── Europe/
│       ├── Eurostat/     # European statistics
│       ├── ECB/          # European Central Bank data
│       └── IEA/          # International Energy Agency data
└── Unstructured_Text_Data/ # News archives, policy documents

02_Data_Processed/
├── Carbon_Cleaned_Aligned/  # Cleaned carbon market data
├── Macro_Cleaned_Aligned/   # Cleaned macroeconomic data
├── LLM_Features_Extracted/  # Features extracted from text data
└── Model_Input_Sets/        # Final datasets for DRL training/testing

03_Literature_Review/
├── Reference_Paper_S65/     # Notes and related work
├── PSO_VMD_Papers/         # PSO and VMD related papers
├── DRL_Trading_Papers/     # DRL trading papers
├── LLM_Finance_Papers/     # LLM in finance papers
└── Carbon_Market_Analysis_Papers/

04_Code/
├── Data_Collection_APIs/    # Data download/access scripts
├── Preprocessing_Cleaning/  # Data validation, cleaning, alignment
├── PSO_VMD_Implementation/  # VMD decomposition and PSO optimization
├── DRL_Agent/
│   ├── Environments/       # Custom Gym environments
│   ├── Models/            # DRL model architectures
│   ├── Training_Scripts/  # Training scripts
│   └── Evaluation_Backtesting_Scripts/
├── LLM_Pipeline/
│   ├── Text_Acquisition/
│   ├── Sentiment_Analysis_Scripts/
│   ├── Topic_Modeling_Scripts/
│   └── Feature_Integration_Scripts/
└── Utilities/              # Common functions, plotting tools

05_Results/
├── Exploratory_Data_Analysis/
├── PSO_VMD_Outputs/
├── DRL_Baseline_Performance/
├── DRL_LLM_Performance/
├── Comparative_Analysis/
└── Log_Files/

06_Manuscript/
├── Drafts/
├── Figures_Tables_for_Paper/
└── Journal_Submission_Materials/
```

## Project Description
This project focuses on developing a deep reinforcement learning (DRL) based trading system for carbon markets, incorporating both structured market data and unstructured text data through LLM-based feature extraction.

## Key Components
1. Data Collection and Processing
2. PSO-VMD Implementation for Time Series Decomposition
3. DRL Agent Development
4. LLM Pipeline for Text Analysis
5. Model Training and Evaluation
6. Results Analysis and Documentation 