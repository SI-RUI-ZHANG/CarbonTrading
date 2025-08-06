from pathlib import Path
import shutil
import os
import glob

CN2EN = {
    "中国": "China",
    "湖北": "Hubei",
    "广东": "Guangdong",
    "用电量": "ElectricityConsumption",
    "工业增加值": "IndustrialAddedValue",
    "可比价": "RealPrices",
    "规模以上工业企业": "AboveScaleIndustry",
    "全社会用电量": "TotalElectricityConsumption",
    "发电量": "ElectricityGeneration",
    "GDP": "GDP",
    "累计值": "Cumulative",
    "当月值": "Monthly",
    "当月同比": "YoY",
    "产量": "Output",
    "粗钢": "CrudeSteel",
    "原油加工量": "CrudeOilProcessing",
    "原煤": "RawCoal",
    "水泥": "Cement",
    "社会融资规模": "TotalSocialFinancing",
    "火电": "ThermalPower",
    "制造业PMI": "ManufacturingPMI",
    "CPI": "CPI",
    "CFETS": "CFETS",
    "即期汇率": "SpotFX",
    "美元兑人民币": "USD_CNY",
    "欧洲ARA港": "ARA_Europe",
    "现价": "CurrentPrices",
    "当月值": "Monthly",
    "动力煤": "ThermalCoal",
    "现货价": "SpotPrice",
    "期货收盘价(连续)": "FuturesClose(Cont)",
    "期货结算价(连续)": "FuturesSettle(Cont)",
    "NYMEX天然气": "NYMEX_NatGas",
    "欧盟排放配额(EUA)": "EUA_Futures",
    "布伦特原油": "BrentCrude",
}

REGION_MAP = {
    "湖北": "hubei",
    "广东": "guangdong",
}


# -------------------------------------------------------------------
# 2.  Helpers
# -------------------------------------------------------------------
def translate_piece(piece: str) -> str:
    """Translate a single underscore-delimited Chinese piece."""
    return CN2EN.get(piece, piece)  # fallback = keep original


def translate_filename(stem: str) -> str:
    """stem without extension → english_stem"""
    parts = stem.split("_")
    translated = [translate_piece(p) for p in parts]
    return "_".join(translated)


def detect_region(first_piece: str) -> str:
    """Return folder name for region."""
    return REGION_MAP.get(first_piece, "national_or_global")


# -------------------------------------------------------------------
# 3.  Main routine
# -------------------------------------------------------------------
def process_folder(src_dir: Path, dst_dir: Path, dry: bool = False):
    parquet_paths = src_dir.glob("*.parquet")  # Changed from rglob to glob - only root level
    for path in parquet_paths:
        stem_cn = path.stem  # without .parquet
        pieces = stem_cn.split("_")
        region_folder = detect_region(pieces[0])
        stem_en = translate_filename(stem_cn)

        target_dir = dst_dir / region_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{stem_en}.parquet"

        if dry:
            print(f"[dry-run] {path}  →  {target_path}")
        else:
            shutil.copy2(path, target_path)
            print(f"copied {path.name}  →  {target_path.relative_to(dst_dir)}")


# Parse all macro files
base_dir = os.path.dirname(os.path.abspath(__file__))
path_macro = os.path.join(base_dir, "../../02_Data_Processed/02_Macroeconomic_Indicators/03_Forward_Filled_Daily/")
src_dir = Path(path_macro)
dst_dir = Path(path_macro) 
process_folder(src_dir, dst_dir, dry=False)
    
print("Done")