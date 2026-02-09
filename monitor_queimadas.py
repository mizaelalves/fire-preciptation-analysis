#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
App em Python (sem Streamlit) que:
- Lê um shapefile (polígono(s) de interesse)
- Busca focos de incêndio de um mês específico (ou o atual) via NASA FIRMS API (VIIRS/MODIS, NRT)
- Recorta espacialmente os pontos pelo shapefile
- Imprime o CSV filtrado no STDOUT

Uso:
    # Defina sua FIRMS MAP_KEY (gratuita) como variável de ambiente:
    export FIRMS_MAP_KEY="SUA_CHAVE_AQUI"

    # Mês atual:
    python focos_firms_mes_atual.py caminho/para/regiao.shp > focos_filtrados.csv

    # Mês específico (ex: Setembro de 2023):
    python focos_firms_mes_atual.py caminho/para/regiao.shp 2023-09 > focos_filtrados.csv

Dependências:
    pip install geopandas pandas shapely requests pyproj fiona python-dateutil

Observações:
- A FIRMS API permite consultar no máximo 10 dias por requisição; o script percorre o mês em blocos.
- Fontes consultadas: VIIRS (NPP, NOAA20, NOAA21) e MODIS (NRT). Você pode ajustar a lista SOURCES abaixo.
"""

import os
import sys
import io
import datetime as dt
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Optional

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# Produtos NRT (Near Real-Time)
SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
]
DAY_RANGE_MAX = 10  # limite da API


def first_day_of_month(d: dt.date) -> dt.date:
    return d.replace(day=1)


def last_day_of_month(d: dt.date) -> dt.date:
    return (first_day_of_month(d) + relativedelta(months=1)) - relativedelta(days=1)


def chunk_dates(start: dt.date, end: dt.date, step_days: int = DAY_RANGE_MAX) -> List[Tuple[dt.date, int]]:
    """
    Divide [start, end] em janelas de até step_days.
    Retorna lista de pares (data_inicial, day_range) conforme a API (range inclui o dia inicial).
    """
    out = []
    cur = start
    while cur <= end:
        rem = (end - cur).days + 1
        span = min(step_days, rem)
        out.append((cur, span))
        cur = cur + dt.timedelta(days=span)
    return out


def read_shapefile_to_polygon(shp_path: str) -> gpd.GeoSeries:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise RuntimeError("Shapefile sem feições.")
    if gdf.crs is None:
        raise RuntimeError("Shapefile sem CRS definido. Defina o CRS correto (ex.: EPSG:4326).")
    if str(gdf.crs).lower().replace(" ", "") != "epsg:4326":
        gdf = gdf.to_crs(epsg=4326)
    geom = unary_union(gdf.geometry)
    return gpd.GeoSeries([geom], crs="EPSG:4326")


def area_bbox_from_polygon(poly_series: gpd.GeoSeries) -> str:
    minx, miny, maxx, maxy = poly_series.total_bounds
    # Clampa para limites do serviço
    minx = max(minx, -180.0)
    maxx = min(maxx, 180.0)
    miny = max(miny, -90.0)
    maxy = min(maxy, 90.0)
    return f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}"


def fetch_firms_csv(map_key: str, source: str, bbox: str, day_range: int, start_date: Optional[dt.date] = None) -> Optional[pd.DataFrame]:
    """
    Chama a API de área da FIRMS e retorna DataFrame (ou None se sem dados).
    Endpoint:
      /api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE][/YYYY-MM-DD]
    """
    if start_date:
        url = f"{FIRMS_API_BASE}/{map_key}/{source}/{bbox}/{day_range}/{start_date.strftime('%Y-%m-%d')}"
    else:
        url = f"{FIRMS_API_BASE}/{map_key}/{source}/{bbox}/{day_range}"

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    # Quando não há dados, o serviço pode retornar texto curto ou só cabeçalho
    if not text or "\n" not in text:
        return None
    # Alguns casos podem retornar mensagem de erro no corpo
    if text.lower().startswith("error") or "invalid" in text.lower():
        raise RuntimeError(f"Erro da FIRMS em {source}: {text[:200]}")
    df = pd.read_csv(io.StringIO(text))
    if df.empty or df.shape[0] == 0:
        return None
    # Normaliza nomes de colunas comuns
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns={k: v for k, v in lower_map.items()}, inplace=True)
    return df


def lat_lon_columns(df: pd.DataFrame) -> Tuple[str, str]:
    for lat_cand in ["latitude", "lat"]:
        if lat_cand in df.columns:
            lat_col = lat_cand
            break
    else:
        raise RuntimeError("Coluna de latitude não encontrada no retorno da FIRMS.")
    for lon_cand in ["longitude", "lon"]:
        if lon_cand in df.columns:
            lon_col = lon_cand
            break
    else:
        raise RuntimeError("Coluna de longitude não encontrada no retorno da FIRMS.")
    return lat_col, lon_col


def to_points_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    lat_col, lon_col = lat_lon_columns(df)

    def to_float(x):
        try:
            return float(str(x).replace(",", ".")) if pd.notnull(x) else None
        except Exception:
            return None

    lat = df[lat_col].map(to_float)
    lon = df[lon_col].map(to_float)
    mask = lat.notnull() & lon.notnull()
    if not mask.any():
        return gpd.GeoDataFrame(columns=list(df.columns) + ["geometry"], crs="EPSG:4326")
    pts = [Point(xy) for xy in zip(lon[mask], lat[mask])]
    gdf = gpd.GeoDataFrame(df.loc[mask].copy(), geometry=pts, crs="EPSG:4326")
    return gdf


def spatial_subset(df: pd.DataFrame, region_poly: gpd.GeoSeries) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[])
    gdf = to_points_gdf(df)
    if gdf.empty:
        return pd.DataFrame(columns=list(df.columns))
    region = region_poly.iloc[0]
    # filtro rápido por bbox
    minx, miny, maxx, maxy = region.bounds
    bbox_mask = gdf.geometry.bounds.apply(
        lambda row: (minx <= row["minx"] <= maxx or minx <= row["maxx"] <= maxx) and (miny <= row["miny"] <= maxy or miny <= row["maxy"] <= maxy),
        axis=1,
    )
    gdf = gdf.loc[bbox_mask] if bbox_mask.any() else gdf
    inside = gdf[gdf.geometry.within(region.buffer(0))]
    return pd.DataFrame(inside.drop(columns=["geometry"]))


def main():
    if len(sys.argv) < 2:
        print("Uso: python focos_firms_mes_atual.py caminho/para/regiao.shp [AAAA-MM] > focos_filtrados.csv", file=sys.stderr)
        sys.exit(1)

    shp_path = sys.argv[1]
    year_month_str = sys.argv[2] if len(sys.argv) > 2 else None

    map_key = '703a103baad106d0baa380f10a209b23'
    if not map_key:
        print("Erro: defina a variável de ambiente FIRMS_MAP_KEY com a sua chave da FIRMS API.", file=sys.stderr)
        sys.exit(2)

    today = dt.date.today()
    target_month_date = None

    if year_month_str:
        try:
            target_month_date = dt.datetime.strptime(f"{year_month_str}-01", "%Y-%m-%d").date()
        except ValueError:
            print(f"Erro: Formato de data inválido '{year_month_str}'. Use AAAA-MM.", file=sys.stderr)
            sys.exit(4)
    else:
        target_month_date = today

    start = first_day_of_month(target_month_date)
    end = last_day_of_month(target_month_date)

    # Se o mês alvo for o mês corrente, o fim é hoje. Senão, é o último dia do mês.
    if start.year == today.year and start.month == today.month:
        end = today

    print(f"Buscando focos de incêndio para o período de {start.strftime('%Y-%m-%d')} a {end.strftime('%Y-%m-%d')}", file=sys.stderr)

    try:
        region_poly = read_shapefile_to_polygon(shp_path)
        bbox = area_bbox_from_polygon(region_poly)

        # Coleta de dados mês atual por blocos de até 10 dias e por fonte
        dfs_all: List[pd.DataFrame] = []
        for source in SOURCES:
            for start_chunk, day_range in chunk_dates(start, end, DAY_RANGE_MAX):
                df_chunk = fetch_firms_csv(map_key, source, bbox, day_range, start_chunk)
                if df_chunk is not None and not df_chunk.empty:
                    # Adiciona metadados do produto
                    df_chunk["firms_source"] = source
                    dfs_all.append(df_chunk)

        if not dfs_all:
            # Nenhum dado bruto no mês / área; imprime CSV vazio com cabeçalho mínimo
            empty_df = pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "confidence", "bright_ti4", "bright_ti5", "frp", "firms_source"])
            empty_df.to_csv(sys.stdout, index=False)
            return

        raw = pd.concat(dfs_all, ignore_index=True)
        # recorte espacial
        subset = spatial_subset(raw, region_poly)

        if subset.empty:
            # mesmo tratamento, mas com as colunas disponíveis
            cols = list(dict.fromkeys(["latitude", "longitude"] + list(raw.columns)))
            pd.DataFrame(columns=cols).to_csv(sys.stdout, index=False)
            return

        # Ordena e remove duplicatas conservadoramente
        sort_cols = [c for c in ["acq_date", "acq_time", "firms_source", "latitude", "longitude"] if c in subset.columns]
        subset.sort_values(by=sort_cols, inplace=True, ignore_index=True)
        subset.drop_duplicates(inplace=True, ignore_index=True)

        # Emite CSV final
        subset.to_csv(sys.stdout, index=False)

    except requests.HTTPError as e:
        print(f"Erro HTTP na FIRMS API: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
