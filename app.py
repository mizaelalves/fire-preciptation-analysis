# app.py
# App Streamlit: chuva por ponto no mapa + COMPARATIVO MENSAL
# - Clique no mapa para escolher latitude/longitude
# - Gráficos (diário, acumulado, janela 7d)
# - Comparativo mensal entre dois meses (mesmos "dias do mês")
# - Download do CSV
#
# Execução:
#   1. Crie um arquivo .env na raiz do projeto:
#      FIRMS_MAP_KEY="SUA_CHAVE_API_DA_FIRMS_AQUI"
#   2. Instale as dependências:
#      pip install streamlit pandas requests streamlit-folium folium altair python-dateutil geopandas shapely python-dotenv
#   3. Execute o app:
#      streamlit run app.py
#
# Dados: NASA POWER (PRECTOTCORR, mm/dia)

import os
import datetime as dt
import io
from urllib.parse import urlencode
from dotenv import load_dotenv
import re

# Carrega variáveis de ambiente do arquivo .env no início do script
load_dotenv()

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import shape


# =========================
# Config & Helpers
# =========================
st.set_page_config(page_title="Análise de Chuva e Queimadas", page_icon="🌧️", layout="wide")
st.title("🌧️ Análise de Chuva e Queimadas")
st.caption("NASA POWER • PRECTOTCORR (mm/dia)")

# Estado inicial (São Paulo como exemplo)
if "lat" not in st.session_state:
    st.session_state.lat = -23.5505
if "lon" not in st.session_state:
    st.session_state.lon = -46.6333
if "region_gdf" not in st.session_state:
    st.session_state.region_gdf = None


def first_day_of_month(date: dt.date) -> dt.date:
    return date.replace(day=1)

def last_day_of_month(date: dt.date) -> dt.date:
    return (first_day_of_month(date) + relativedelta(months=1)) - relativedelta(days=1)

def today_local() -> dt.date:
    return dt.date.today()

def ensure_month_bounds(some_day: dt.date, today: dt.date) -> tuple[dt.date, dt.date]:
    """Retorna (start, end) do mês de some_day, limitando end a 'today' se for o mês corrente."""
    start = first_day_of_month(some_day)
    end = min(last_day_of_month(some_day), today)
    return start, end

def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    out = df.copy()
    if "date" in out.columns:
        # Extrai a data se for datetime
        if pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["date"] = out["date"].dt.date
        out["date"] = out["date"].astype(str)
    out.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def sanitize_filename(name: str) -> str:
    """Sanitiza uma string para ser usada como um nome de arquivo seguro."""
    if not name:
        return ""
    name = name.strip().lower()
    name = re.sub(r'[\s_]+', '-', name)  # Substitui espaços e underscores por hífens
    name = re.sub(r'[^\w-]', '', name)   # Remove caracteres não alfanuméricos (exceto hífens)
    return name

@st.cache_data(show_spinner="Lendo arquivo geoespacial...")
def read_geospatial_file(uploaded_file) -> gpd.GeoDataFrame | None:
    """Lê um arquivo .zip (shapefile) ou .geojson e retorna um GeoDataFrame."""
    if uploaded_file is None:
        return None
    
    fname = uploaded_file.name
    if fname.endswith(".zip"):
        # Usa vfs para ler o shapefile de dentro do zip
        gdf = gpd.read_file(f"zip://{fname}", vfs=f"zip://{uploaded_file.read()}")
    elif fname.endswith(".geojson"):
        gdf = gpd.read_file(uploaded_file)
    else:
        st.error("Formato de arquivo não suportado. Use .zip para shapefile ou .geojson.")
        return None

    # Garante CRS WGS84
    if gdf.crs is None:
        st.warning("O arquivo não tem um CRS definido. Assumindo WGS84 (EPSG:4326).")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    
    return gdf


# =========================
# Dados (NASA FIRMS)
# =========================
FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
SOURCES = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT"]
DAY_RANGE_MAX = 10  # limite da API

def chunk_dates(start: dt.date, end: dt.date, step_days: int = DAY_RANGE_MAX):
    """Divide [start, end] em janelas de até step_days."""
    cur = start
    while cur <= end:
        rem = (end - cur).days + 1
        span = min(step_days, rem)
        yield (cur, span)
        cur = cur + dt.timedelta(days=span)

def area_bbox_from_gdf(gdf: gpd.GeoDataFrame) -> str:
    """Calcula BBOX no formato da API a partir de um GeoDataFrame."""
    minx, miny, maxx, maxy = gdf.total_bounds
    minx = max(minx, -180.0); maxx = min(maxx, 180.0)
    miny = max(miny, -90.0); maxy = min(maxy, 90.0)
    return f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}"

@st.cache_data(show_spinner="Buscando focos de queimada (FIRMS)...", ttl=3600)
def fetch_firms_data(
    map_key: str, 
    _region_gdf: gpd.GeoDataFrame, 
    start_date: dt.date, 
    end_date: dt.date
) -> dict:
    """Busca dados da FIRMS para uma área e período, retorna dict com dados brutos e filtrados."""
    bbox = area_bbox_from_gdf(_region_gdf)
    dfs_all = []

    for source in SOURCES:
        for start_chunk, day_range in chunk_dates(start_date, end_date, DAY_RANGE_MAX):
            try:
                url = f"{FIRMS_API_BASE}/{map_key}/{source}/{bbox}/{day_range}/{start_chunk.strftime('%Y-%m-%d')}"
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                text = r.text.strip()
                if not text or "\n" not in text:
                    continue
                if "error" in text.lower() or "invalid" in text.lower():
                    st.warning(f"API FIRMS retornou aviso para '{source}': {text[:100]}")
                    continue
                
                df_chunk = pd.read_csv(io.StringIO(text))
                if not df_chunk.empty:
                    df_chunk["firms_source"] = source
                    dfs_all.append(df_chunk)

            except requests.HTTPError as e:
                # Silencioso se for 404 (sem dados), erro para outros
                if e.response.status_code != 404:
                    st.error(f"Erro HTTP na API FIRMS ({source}): {e}")
                continue # Pula para próxima fonte/data
            except Exception as e:
                st.error(f"Falha ao processar dados da FIRMS ({source}): {e}")
                continue
    
    empty_result = {"raw": pd.DataFrame(), "filtered": pd.DataFrame()}
    if not dfs_all:
        return empty_result

    raw_df = pd.concat(dfs_all, ignore_index=True)
    
    # Normaliza nomes de colunas
    raw_df.rename(columns={c: c.lower() for c in raw_df.columns}, inplace=True)
    
    # Filtro espacial preciso (dentro do polígono)
    if "latitude" not in raw_df.columns or "longitude" not in raw_df.columns:
        st.error("Resposta da FIRMS não contém colunas 'latitude' ou 'longitude'.")
        return empty_result # Retorna DF vazio se não houver coordenadas

    pts = gpd.points_from_xy(raw_df["longitude"], raw_df["latitude"], crs="EPSG:4326")
    gdf_pts = gpd.GeoDataFrame(raw_df.copy(), geometry=pts)
    
    region_geom = _region_gdf.unary_union
    inside_mask = gdf_pts.within(region_geom)
    
    result_gdf = gdf_pts.loc[inside_mask].copy()
    
    # Limpa e ordena
    if not result_gdf.empty:
        sort_cols = [c for c in ["acq_date", "acq_time"] if c in result_gdf.columns]
        if sort_cols:
            result_gdf.sort_values(by=sort_cols, inplace=True, ignore_index=True)
        result_gdf.drop_duplicates(inplace=True, ignore_index=True)

    filtered_df = pd.DataFrame(result_gdf.drop(columns=["geometry"]))
    return {"raw": raw_df, "filtered": filtered_df}


def estimar_area_queimada(firms_df: pd.DataFrame) -> float:
    """
    Estima a área queimada em hectares a partir de um DataFrame de focos da FIRMS.
    Assume uma área fixa por tipo de satélite/instrumento (pixel).
    VIIRS: 375m x 375m = 14.0625 ha
    MODIS: 1km x 1km = 100 ha
    Esta é uma aproximação e não um dado oficial de área queimada.
    """
    if firms_df is None or firms_df.empty:
        return 0.0

    # Heurística para determinar a área do pixel com base na fonte
    def get_area_ha(source: str) -> float:
        if pd.isna(source):
            return 0.0
        s = source.upper()
        if "VIIRS" in s:
            return 14.0625  # 375m * 375m
        elif "MODIS" in s:
            return 100.0  # 1km * 1km
        else:
            return 0.0 # Fonte desconhecida, não soma

    if "firms_source" in firms_df.columns:
        total_area_ha = firms_df["firms_source"].apply(get_area_ha).sum()
    else:
        total_area_ha = 0.0
    
    return total_area_ha


# =========================
# Dados (NASA POWER)
# =========================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_daily_rain_mm(lat: float, lon: float, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Retorna DataFrame com colunas:
    - date (datetime.date)
    - precip_mm (float, NaN quando sem dado)
    """
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR",
        "community": "RE",
        "longitude": f"{lon:.6f}",
        "latitude": f"{lat:.6f}",
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    url = f"{base}?{urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    try:
        series = data["properties"]["parameter"]["PRECTOTCORR"]
    except KeyError:
        raise RuntimeError("Estrutura inesperada na resposta da API NASA POWER.")

    rows = []
    for yyyymmdd, val in sorted(series.items()):
        v = float(val) if val is not None else float("nan")
        v = float("nan") if v <= -900 else v  # missing
        rows.append({
            "date": dt.datetime.strptime(yyyymmdd, "%Y%m%d").date(),
            "precip_mm": v
        })
    df = pd.DataFrame(rows)
    return df


# =========================
# Nova Função: Comparativo Mensal
# =========================
def comparar_meses(lat: float, lon: float, mes1: dt.date, mes2: dt.date, hoje: dt.date) -> dict:
    """
    Compara dois meses para um ponto (lat, lon).
    Retorna dict com:
      - df1, df2: séries diárias (date, precip_mm)
      - df_comp: long format com colunas [day, precip_mm, mes_label]
      - metrica1, metrica2: resumo de cada mês
    """
    s1, e1 = ensure_month_bounds(mes1, hoje)
    s2, e2 = ensure_month_bounds(mes2, hoje)

    df1 = fetch_daily_rain_mm(lat, lon, s1, e1).copy()
    df2 = fetch_daily_rain_mm(lat, lon, s2, e2).copy()

    # Campos auxiliares
    for df in (df1, df2):
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.day
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")

    label1 = s1.strftime("%b/%Y")
    label2 = s2.strftime("%b/%Y")

    df1["mes_label"] = label1
    df2["mes_label"] = label2

    # Junta em formato longo para gráficos: alinhar por "dia do mês"
    max_day = max(df1["day"].max(), df2["day"].max())
    all_days = pd.DataFrame({"day": range(1, int(max_day) + 1)})
    df1m = all_days.merge(df1[["day", "precip_mm"]], on="day", how="left")
    df1m["mes_label"] = label1
    df2m = all_days.merge(df2[["day", "precip_mm"]], on="day", how="left")
    df2m["mes_label"] = label2
    df_comp = pd.concat([df1m, df2m], ignore_index=True)

    # Métricas
    def resumo(df, start, end):
        total = float(df["precip_mm"].sum(skipna=True))
        dias_chuva = int((df["precip_mm"].fillna(0) > 0).sum())
        max_diaria = float(df["precip_mm"].max(skipna=True))
        media_diaria = float(df["precip_mm"].mean(skipna=True))
        return {
            "periodo": f"{start.strftime('%d/%m/%Y')} → {end.strftime('%d/%m/%Y')}",
            "total_mm": total,
            "dias_chuva": dias_chuva,
            "max_diaria_mm": max_diaria,
            "media_diaria_mm": media_diaria,
        }

    metrica1 = resumo(df1, s1, e1)
    metrica2 = resumo(df2, s2, e2)

    return {
        "df1": df1,
        "df2": df2,
        "df_comp": df_comp,  # colunas: day, precip_mm, mes_label
        "label1": label1,
        "label2": label2,
        "metrica1": metrica1,
        "metrica2": metrica2,
        "periodo1": (s1, e1),
        "periodo2": (s2, e2),
    }


# =========================
# Funções de Geração de Componentes Visuais
# =========================
def display_rain_analysis(rain_df: pd.DataFrame, start_date: dt.date, end_date: dt.date, area_name: str):
    """Exibe métricas e gráficos para um DataFrame de chuva."""
    if rain_df.empty:
        st.warning("Nenhum dado de precipitação retornado para o período.")
        return
    
    st.markdown(f"**Período:** {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")
    total = float(rain_df["precip_mm"].sum(skipna=True))
    dias_chuva = int((rain_df["precip_mm"].fillna(0) > 0).sum())
    max_diaria = float(rain_df["precip_mm"].max(skipna=True))

    m1, m2, m3 = st.columns(3)
    m1.metric("Total no período (mm)", f"{total:.1f}")
    m2.metric("Dias com chuva", f"{dias_chuva}")
    m3.metric("Máxima diária (mm)", f"{max_diaria:.1f}")

    df_plot = rain_df.copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])
    df_plot["precip_mm"] = pd.to_numeric(df_plot["precip_mm"], errors="coerce")

    st.altair_chart(
        alt.Chart(df_plot, title="Chuva diária (mm)")
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Data"),
            y=alt.Y("precip_mm:Q", title="Chuva Diária (mm)"),
            tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("precip_mm:Q", title="mm")]
        ),
        use_container_width=True
    )
    
    with st.expander("Ver tabela de dados de precipitação"):
        st.dataframe(rain_df, use_container_width=True, hide_index=True)
        fname_part = sanitize_filename(area_name) or f"{st.session_state.lat:.4f}_{st.session_state.lon:.4f}"
        st.download_button(
            "⬇️ Baixar CSV", data=csv_bytes(rain_df),
            file_name=f"precipitacao_{fname_part}_{start_date.strftime('%Y%m')}.csv", mime="text/csv"
        )

def display_firms_analysis(firms_data: dict, start_date: dt.date, end_date: dt.date, region_gdf: gpd.GeoDataFrame, area_name: str):
    """Exibe métricas, mapa e gráficos para um DataFrame de queimadas."""
    firms_df = firms_data.get("filtered", pd.DataFrame())
    firms_df_raw = firms_data.get("raw", pd.DataFrame())

    if firms_df.empty:
        st.success("✅ Nenhum foco de queimada detectado na área e período selecionados.")
        if not firms_df_raw.empty:
            st.info(f"Foram encontrados {len(firms_df_raw)} focos na área de busca inicial (bounding box), mas nenhum dentro do polígono exato.")
            with st.expander("Ver dados brutos (antes do filtro espacial)"):
                st.dataframe(firms_df_raw, use_container_width=True, hide_index=True)
                fname_part = sanitize_filename(area_name) or "area-selecionada"
                st.download_button(
                    "⬇️ Baixar CSV Bruto (sem filtro espacial)", data=csv_bytes(firms_df_raw),
                    file_name=f"queimadas_bruto_{fname_part}_{start_date.strftime('%Y%m')}.csv", mime="text/csv"
                )
        return

    st.markdown(f"**Período:** {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")
    total_focos = len(firms_df)
    fontes = firms_df["firms_source"].nunique()
    
    m1, m2 = st.columns(2)
    m1.metric("Total de focos de queimada", f"{total_focos}")
    m2.metric("Fontes de satélite", f"{fontes}")

    # Mapa
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=9, control_scale=True)
    folium.GeoJson(region_gdf, style_function=lambda x: {"fillColor": "orange", "color": "black", "weight": 2, "fillOpacity": 0.2}).add_to(m)
    locations = firms_df[['latitude', 'longitude']].values.tolist()
    popups = [f"Data: {row['acq_date']}<br>Fonte: {row['firms_source']}" for _, row in firms_df.iterrows()]
    MarkerCluster(locations=locations, popups=popups).add_to(m)
    st_folium(m, width=None, height=400, returned_objects=[])

    # Gráfico de barras diário
    daily_counts = firms_df.groupby("acq_date").size().reset_index(name="count")
    daily_counts["acq_date"] = pd.to_datetime(daily_counts["acq_date"])
    st.altair_chart(
        alt.Chart(daily_counts, title="Número de focos por dia")
        .mark_bar(color="orange").encode(
            x=alt.X("acq_date:T", title="Data"), y=alt.Y("count:Q", title="Nº de Focos"),
            tooltip=[alt.Tooltip("acq_date:T", title="Data"), alt.Tooltip("count:Q", title="Focos")]
        ), use_container_width=True
    )

    with st.expander("Ver tabela de dados de queimadas"):
        st.dataframe(firms_df, use_container_width=True, hide_index=True)
        fname_part = sanitize_filename(area_name) or "area-selecionada"
        st.download_button(
            "⬇️ Baixar CSV Filtrado", data=csv_bytes(firms_df),
            file_name=f"queimadas_{fname_part}_{start_date.strftime('%Y%m')}.csv", mime="text/csv"
        )
        if not firms_df_raw.empty:
            st.download_button(
                "⬇️ Baixar CSV Bruto (sem filtro espacial)", data=csv_bytes(firms_df_raw),
                file_name=f"queimadas_bruto_{fname_part}_{start_date.strftime('%Y%m')}.csv", mime="text/csv",
                key="download_raw_firms"
            )


def display_single_year_firms_detail(firms_data: dict, year: str, region_gdf: gpd.GeoDataFrame, area_name: str):
    """Exibe uma análise detalhada de queimadas para um único ano, com detalhamento mensal."""
    firms_df = firms_data.get("filtered", pd.DataFrame())
    firms_df_raw = firms_data.get("raw", pd.DataFrame())

    if firms_df.empty:
        st.success(f"✅ Nenhum foco de queimada detectado na área para o ano de {year}.")
        if not firms_df_raw.empty:
            st.info(f"Foram encontrados {len(firms_df_raw)} focos na área de busca inicial (bounding box), mas nenhum dentro do polígono exato.")
            with st.expander("Ver dados brutos do ano (antes do filtro espacial)"):
                st.dataframe(firms_df_raw, use_container_width=True, hide_index=True)
                fname_part = sanitize_filename(area_name) or "area-selecionada"
                st.download_button(
                    "⬇️ Baixar CSV Bruto do Ano Completo", data=csv_bytes(firms_df_raw),
                    file_name=f"queimadas_bruto_{fname_part}_{year}.csv", mime="text/csv"
                )
        return

    st.markdown(f"### Análise Detalhada de Queimadas para {year}")

    # 1. Métricas anuais
    total_focos_ano = len(firms_df)
    total_area_ano = estimar_area_queimada(firms_df)
    
    m1, m2 = st.columns(2)
    m1.metric("Total de Focos no Ano", f"{total_focos_ano}")
    m2.metric("Área Queimada Estimada (ha)", f"{total_area_ano:,.2f}")
    st.caption("ℹ️ A área queimada é uma **estimativa** baseada na área do pixel do satélite (VIIRS: ~14ha, MODIS: ~100ha).")

    # 2. Processamento mensal
    df = firms_df.copy()
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['month'] = df['acq_date'].dt.month

    monthly_summary = []
    for month_num in range(1, 13):
        month_df = df[df['month'] == month_num]
        focos = len(month_df)
        area_ha = estimar_area_queimada(month_df)
        monthly_summary.append({
            "Mês": dt.date(int(year), month_num, 1).strftime("%b"),
            "Nº de Focos": focos,
            "Área Estimada (ha)": area_ha
        })
    
    summary_df = pd.DataFrame(monthly_summary)
    
    # 3. Gráfico e Tabela
    st.divider()
    st.subheader("Distribuição Mensal")
    
    # Gráfico combinado
    base = alt.Chart(summary_df).encode(x=alt.X('Mês:N', sort=None, title='Mês'))
    
    bar_area = base.mark_bar(color='#E6550D').encode(
        y=alt.Y('Área Estimada (ha):Q', title='Área Estimada (ha)', axis=alt.Axis(titleColor='#E6550D')),
        tooltip=[alt.Tooltip("Mês:N"), alt.Tooltip("Área Estimada (ha):Q", format=",.2f")]
    )
    
    line_focos = base.mark_line(color='orange', strokeWidth=2.5, point=True).encode(
        y=alt.Y('Nº de Focos:Q', title='Nº de Focos', axis=alt.Axis(titleColor='orange')),
        tooltip=[alt.Tooltip("Mês:N"), alt.Tooltip("Nº de Focos:Q")]
    )

    st.altair_chart(
        alt.layer(bar_area, line_focos).resolve_scale(y='independent').properties(
            title=f"Focos de Queimada e Área Estimada por Mês em {year}", height=400
        ), use_container_width=True
    )

    with st.expander("Ver tabela de dados mensais"):
        st.dataframe(summary_df.set_index("Mês"), use_container_width=True)

    with st.expander("Ver tabela de todos os focos do ano"):
        st.dataframe(firms_df, use_container_width=True, hide_index=True)
        fname_part = sanitize_filename(area_name) or "area-selecionada"
        st.download_button(
            "⬇️ Baixar CSV Filtrado do Ano Completo", data=csv_bytes(firms_df),
            file_name=f"queimadas_{fname_part}_{year}.csv", mime="text/csv"
        )
        if not firms_df_raw.empty:
            st.download_button(
                "⬇️ Baixar CSV Bruto do Ano Completo", data=csv_bytes(firms_df_raw),
                file_name=f"queimadas_bruto_{fname_part}_{year}.csv", mime="text/csv",
                key="download_raw_year"
            )


def display_combined_analysis(rain_df: pd.DataFrame, firms_data: dict, start_date: dt.date, end_date: dt.date, area_name: str):
    """Exibe gráfico e tabela combinando chuva e queimadas."""
    firms_df = firms_data.get("filtered")
    if rain_df.empty and (firms_df is None or firms_df.empty):
        st.warning("Nenhum dado para análise combinada.")
        return

    # Preparação dos dados
    rain_df["date"] = pd.to_datetime(rain_df["date"])
    rain_df.set_index("date", inplace=True)
    
    if firms_df is not None and not firms_df.empty:
        firms_df["acq_date"] = pd.to_datetime(firms_df["acq_date"])
        daily_counts = firms_df.groupby("acq_date").size().reset_index(name="fire_count")
        daily_counts.rename(columns={"acq_date": "date"}, inplace=True)
        daily_counts.set_index("date", inplace=True)
    else:
        daily_counts = pd.DataFrame(columns=["fire_count"])

    full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    combined_df = pd.DataFrame(index=full_date_range)
    combined_df = combined_df.join(rain_df["precip_mm"]).join(daily_counts["fire_count"])
    combined_df.fillna(0, inplace=True)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={"index": "date"}, inplace=True)

    st.markdown(f"**Período:** {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")

    # Gráfico
    base = alt.Chart(combined_df).encode(x=alt.X('date:T', title='Data'))
    bar_rain = base.mark_bar(color='#5276A7').encode(
        y=alt.Y('precip_mm:Q', title='Precipitação (mm)', axis=alt.Axis(titleColor='#5276A7')),
        tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("precip_mm:Q", title="Chuva (mm)")]
    )
    line_fire = base.mark_line(color='#F46524', strokeWidth=2.5, point=True).encode(
        y=alt.Y('fire_count:Q', title='Focos de Queimada', axis=alt.Axis(titleColor='#F46524')),
        tooltip=[alt.Tooltip("date:T", title="Data"), alt.Tooltip("fire_count:Q", title="Focos")]
    )
    
    st.altair_chart(
        alt.layer(bar_rain, line_fire).resolve_scale(y='independent').properties(
            title="Precipitação Diária vs. Focos de Queimada", height=400
        ), use_container_width=True
    )
    
    with st.expander("Ver tabela de dados combinados"):
        st.dataframe(combined_df, use_container_width=True, hide_index=True)
        fname_part = sanitize_filename(area_name) or "area-selecionada"
        st.download_button(
            "⬇️ Baixar CSV", data=csv_bytes(combined_df),
            file_name=f"chuva_queimadas_{fname_part}_{start_date.strftime('%Y%m')}.csv", mime="text/csv"
        )

def display_annual_rain(analysis_results: list, mes_nome: str):
    """Exibe um gráfico de barras comparando a precipitação total entre períodos."""
    summary_data = []
    for result in analysis_results:
        period_label = result["period"]["label"]
        rain_df = result["rain_df"]
        total_mm = 0
        if rain_df is not None and not rain_df.empty:
            total_mm = rain_df["precip_mm"].sum()
        summary_data.append({"período": period_label, "precip_total_mm": total_mm})
    
    if not summary_data:
        st.warning("Nenhum dado de precipitação encontrado para os períodos anuais.")
        return

    summary_df = pd.DataFrame(summary_data).sort_values("período", ascending=False)
    
    chart = alt.Chart(summary_df, title="Precipitação Total Comparativa").mark_bar().encode(
        x=alt.X("período:N", title="Período", sort=None),
        y=alt.Y("precip_total_mm:Q", title="Precipitação Total (mm)"),
        tooltip=["período", "precip_total_mm"]
    )
    st.altair_chart(chart, use_container_width=True)
    with st.expander("Ver dados anuais de precipitação"):
        st.dataframe(summary_df)

    # --- NOVO: Detalhamento mensal se for ano inteiro ---
    if mes_nome == "Ano Inteiro":
        st.divider()
        st.subheader("Detalhamento Mensal por Ano")

        monthly_data = []
        for result in analysis_results:
            year_label = result["period"]["label"]
            rain_df = result["rain_df"]
            if rain_df is not None and not rain_df.empty:
                df = rain_df.copy()
                df['date'] = pd.to_datetime(df['date'])
                # Agrupa por mês para obter as somas mensais
                monthly_sum = df.set_index('date').resample('M')['precip_mm'].sum().reset_index()
                monthly_sum['year'] = year_label
                monthly_sum['month_num'] = monthly_sum['date'].dt.month
                monthly_sum['month_abbr'] = monthly_sum['date'].dt.strftime('%b')
                monthly_data.append(monthly_sum)

        if not monthly_data:
            st.info("Nenhum dado disponível para o detalhamento mensal.")
            return

        full_monthly_df = pd.concat(monthly_data, ignore_index=True)
        
        # Ordenação correta dos meses no gráfico
        month_order = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

        line_chart = alt.Chart(full_monthly_df, title="Precipitação Mensal Acumulada por Ano").mark_line(point=True).encode(
            x=alt.X('month_abbr:N', title='Mês', sort=month_order),
            y=alt.Y('precip_mm:Q', title='Precipitação Mensal (mm)'),
            color=alt.Color('year:N', title='Ano'),
            tooltip=[
                alt.Tooltip('year', title='Ano'),
                alt.Tooltip('month_abbr', title='Mês'),
                alt.Tooltip('precip_mm:Q', title='Total (mm)', format='.1f')
            ]
        ).properties(height=400)
        
        st.altair_chart(line_chart, use_container_width=True)
        with st.expander("Ver dados mensais detalhados"):
            st.dataframe(full_monthly_df[['year', 'month_abbr', 'precip_mm']], use_container_width=True)


def display_annual_firms(analysis_results: list, mes_nome: str):
    """Exibe métricas e gráficos comparando dados de queimadas anuais."""
    summary_data = []
    all_firms_dfs = []

    for result in analysis_results:
        period_label = result["period"]["label"]
        firms_data = result["firms_data"]
        firms_df = firms_data.get("filtered", pd.DataFrame())
        
        fire_count = 0
        estimated_area = 0.0

        if firms_df is not None and not firms_df.empty:
            fire_count = len(firms_df)
            estimated_area = estimar_area_queimada(firms_df)
            
            df_copy = firms_df.copy()
            df_copy["periodo"] = period_label
            all_firms_dfs.append(df_copy)

        summary_data.append({
            "período": period_label,
            "total_focos": fire_count,
            "area_queimada_ha_estimada": estimated_area
        })
    
    if not any(d['total_focos'] > 0 for d in summary_data):
        st.success("✅ Nenhum foco de queimada detectado nos períodos anuais selecionados.")
        return

    summary_df = pd.DataFrame(summary_data).sort_values("período", ascending=False)
    
    st.dataframe(summary_df.rename(columns={
        "período": "Período",
        "total_focos": "Total de Focos",
        "area_queimada_ha_estimada": "Área Estimada (ha)"
    }).set_index("Período"))
    st.caption("ℹ️ A área queimada é uma **estimativa** baseada na área do pixel do satélite para cada foco detectado (VIIRS: ~14ha, MODIS: ~100ha). Não é um valor oficial.")

    # Gráfico 1: Total de Focos
    chart_focos = alt.Chart(summary_df, title="Total de Focos de Queimada Comparativo").mark_bar(color="orange").encode(
        x=alt.X("período:N", title="Período", sort=None),
        y=alt.Y("total_focos:Q", title="Nº Total de Focos"),
        tooltip=["período", "total_focos"]
    )
    
    # Gráfico 2: Área Estimada
    chart_area = alt.Chart(summary_df, title="Área Queimada Estimada (ha)").mark_bar(color="#E6550D").encode(
        x=alt.X("período:N", title="Período", sort=None),
        y=alt.Y("area_queimada_ha_estimada:Q", title="Área Estimada (ha)"),
        tooltip=["período", alt.Tooltip("area_queimada_ha_estimada:Q", format=".2f")]
    )
    st.altair_chart(chart_focos | chart_area, use_container_width=True)

    # Detalhamento mensal se for ano inteiro
    if mes_nome == "Ano Inteiro" and all_firms_dfs:
        st.divider()
        st.subheader("Detalhamento Mensal de Focos por Ano")
        
        full_firms_df = pd.concat(all_firms_dfs, ignore_index=True)
        full_firms_df['acq_date'] = pd.to_datetime(full_firms_df['acq_date'])
        
        monthly_counts = full_firms_df.set_index('acq_date').groupby('periodo').resample('M').size().reset_index(name='focos')
        monthly_counts['year'] = monthly_counts['periodo']
        monthly_counts['month_num'] = monthly_counts['acq_date'].dt.month
        monthly_counts['month_abbr'] = monthly_counts['acq_date'].dt.strftime('%b')

        month_order = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

        line_chart = alt.Chart(monthly_counts, title="Focos de Queimada Mensais por Ano").mark_line(point=True).encode(
            x=alt.X('month_abbr:N', title='Mês', sort=month_order),
            y=alt.Y('focos:Q', title='Nº de Focos'),
            color=alt.Color('year:N', title='Ano'),
            tooltip=['year', 'month_abbr', 'focos']
        ).properties(height=400)
        
        st.altair_chart(line_chart, use_container_width=True)
        with st.expander("Ver dados mensais detalhados de focos"):
            st.dataframe(monthly_counts[['year', 'month_abbr', 'focos']], use_container_width=True)


def display_annual_combined(analysis_results: list, mes_nome: str):
    """Exibe os dois gráficos de comparação anual."""
    st.subheader("Comparativo Anual: Precipitação")
    display_annual_rain(analysis_results, mes_nome)
    st.divider()
    st.subheader("Comparativo Anual: Focos de Queimada")
    st.info("A análise de queimadas requer o upload de um arquivo de área (Shapefile/GeoJSON).")
    display_annual_firms(analysis_results, mes_nome)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🗺️ 1. Selecione a área")
    
    area_name_in = st.text_input("Nome da área (opcional)", placeholder="Ex: Fazenda Boa Esperança")
    st.session_state.area_name = area_name_in

    input_method = st.radio("Como deseja definir a área?", ["Ponto no mapa", "Upload de Shapefile/GeoJSON"])

    if input_method == "Ponto no mapa":
        st.session_state.region_gdf = None # Limpa região se mudar de método
        st.write("Clique no mapa para escolher a localização. Os campos acompanham o clique.")
        m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=6, control_scale=True)
        folium.TileLayer("OpenStreetMap").add_to(m)
        folium.Marker(
            [st.session_state.lat, st.session_state.lon],
            popup=f"{st.session_state.lat:.5f}, {st.session_state.lon:.5f}",
            draggable=False,
            icon=folium.Icon(icon="cloud", prefix="fa")
        ).add_to(m)

        map_event = st_folium(m, width=None, height=300, returned_objects=["last_clicked"])
        if map_event and map_event.get("last_clicked"):
            st.session_state.lat = float(map_event["last_clicked"]["lat"])
            st.session_state.lon = float(map_event["last_clicked"]["lng"])

        colA, colB = st.columns(2)
        with colA:
            lat_in = st.number_input("Latitude", value=float(st.session_state.lat), step=0.0001, format="%.6f")
        with colB:
            lon_in = st.number_input("Longitude", value=float(st.session_state.lon), step=0.0001, format="%.6f")
        st.session_state.lat = lat_in
        st.session_state.lon = lon_in

    else: # Upload de arquivo
        uploaded_file = st.file_uploader(
            "Envie um arquivo .zip (shapefile) ou .geojson",
            type=["zip", "geojson"],
            accept_multiple_files=False,
        )
        if uploaded_file:
            st.session_state.region_gdf = read_geospatial_file(uploaded_file)
            if st.session_state.region_gdf is not None:
                st.success(f"Área carregada: {len(st.session_state.region_gdf)} polígonos.")
                # Atualiza lat/lon com o centroide para visualização
                centroid = st.session_state.region_gdf.unary_union.centroid
                st.session_state.lat = centroid.y
                st.session_state.lon = centroid.x

    st.divider()
    st.header("🗓️ 2. Período e API")
    
    analysis_mode = st.radio(
        "Selecione o modo de análise",
        ["Mês Único", "Comparativo Mensal", "Análise Anual Específica", "Comparativo Anual"]
    )

    # Pega a chave da variável de ambiente
    firms_api_key = os.getenv("FIRMS_MAP_KEY")
    if not firms_api_key:
        st.warning(
            "A variável de ambiente `FIRMS_MAP_KEY` não está definida. "
            "A análise de queimadas não funcionará."
        )
    else:
        st.success("Chave da API FIRMS carregada da variável de ambiente.")

    today = today_local()
    periods_to_fetch = []

    if analysis_mode == "Mês Único":
        mes_ref = st.date_input(
            "Mês de referência", value=today.replace(day=1),
            min_value=today - relativedelta(years=20), max_value=today,
            help="Escolha qualquer dia dentro do mês de análise."
        )
        start, end = ensure_month_bounds(mes_ref, today)
        periods_to_fetch.append({"start": start, "end": end, "label": start.strftime('%b/%Y')})

    elif analysis_mode == "Comparativo Mensal":
        c1, c2 = st.columns(2)
        with c1:
            mes1 = st.date_input("Mês A", value=today.replace(day=1), min_value=today - relativedelta(years=20), max_value=today)
        with c2:
            mes2 = st.date_input("Mês B", value=(today.replace(day=1) - relativedelta(months=1)), min_value=today - relativedelta(years=20), max_value=today)
        start1, end1 = ensure_month_bounds(mes1, today)
        periods_to_fetch.append({"start": start1, "end": end1, "label": start1.strftime('%b/%Y')})
        start2, end2 = ensure_month_bounds(mes2, today)
        periods_to_fetch.append({"start": start2, "end": end2, "label": start2.strftime('%b/%Y')})

    elif analysis_mode == "Análise Anual Específica":
        anos = list(range(today.year, today.year - 20, -1))
        ano_selecionado = st.selectbox("Selecione o ano para análise", anos)
        start_date = dt.date(ano_selecionado, 1, 1)
        end_date = dt.date(ano_selecionado, 12, 31)
        if end_date > today:
            end_date = today
        periods_to_fetch.append({"start": start_date, "end": end_date, "label": str(ano_selecionado)})

    else:  # Comparativo Anual
        anos = list(range(today.year, today.year - 20, -1))
        ano_ref = st.selectbox("Ano de referência", anos)
        
        meses = {"Ano Inteiro": 0, "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6, "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12}
        mes_nome = st.selectbox("Mês para comparar (ou ano inteiro)", list(meses.keys()))
        mes_num = meses[mes_nome]

        num_anos = st.slider("Número de anos para comparar (incluindo o de referência)", min_value=2, max_value=10, value=5)

        for i in range(num_anos):
            year = ano_ref - i
            label, start_date, end_date = "", None, None
            
            if mes_num == 0:  # Ano Inteiro
                start_date = dt.date(year, 1, 1)
                end_date = dt.date(year, 12, 31)
                label = str(year)
            else:
                if year == today.year and mes_num > today.month: 
                    continue
                start_date = dt.date(year, mes_num, 1)
                end_date = last_day_of_month(start_date)
                label = start_date.strftime('%b/%Y')
            
            if end_date > today: 
                end_date = today
            periods_to_fetch.append({"start": start_date, "end": end_date, "label": label})
    
    fetch_btn = st.button("🔎 Analisar período", use_container_width=True, type="primary")

# =========================
# Corpo Principal do App
# =========================
st.markdown(f"**Localização selecionada:**")

area_name = st.session_state.get("area_name", "")
if area_name:
    st.markdown(f"### Análise para a área: **{area_name}**")

if st.session_state.region_gdf is not None:
    st.markdown(f"Região definida por arquivo. Centroide em: `{st.session_state.lat:.5f}, {st.session_state.lon:.5f}`")
else:
    st.markdown(f"Ponto: `{st.session_state.lat:.5f}, {st.session_state.lon:.5f}`")


if fetch_btn:
    analysis_results = []
    try:
        for period in periods_to_fetch:
            with st.spinner(f"Buscando dados para {period['label']}..."):
                # Precipitação é sempre buscada
                rain_df = fetch_daily_rain_mm(st.session_state.lat, st.session_state.lon, period['start'], period['end'])
                
                firms_data = {"raw": pd.DataFrame(), "filtered": pd.DataFrame()}
                # Queimadas são buscadas se uma região for fornecida via arquivo
                if st.session_state.region_gdf is not None and firms_api_key:
                    firms_data = fetch_firms_data(firms_api_key, st.session_state.region_gdf, period['start'], period['end'])
                
                analysis_results.append({"rain_df": rain_df, "firms_data": firms_data, "period": period})
    except Exception as e:
        st.error(f"Ocorreu um erro durante a busca de dados: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["💧 Precipitação", "🔥 Queimadas", "📈 Combinado"])

    # --- ABA 1: PRECIPITAÇÃO ---
    with tab1:
        if analysis_mode in ["Mês Único", "Análise Anual Específica"]:
            res = analysis_results[0]
            display_rain_analysis(res["rain_df"], res["period"]["start"], res["period"]["end"], area_name)
        
        elif analysis_mode == "Comparativo Mensal":
            st.header("Comparativo de Precipitação")
            c1, c2 = st.columns(2)
            res1, res2 = analysis_results[0], analysis_results[1]
            with c1:
                st.subheader(f"Mês A: {res1['period']['label']}")
                display_rain_analysis(res1["rain_df"], res1["period"]["start"], res1["period"]["end"], area_name)
            with c2:
                st.subheader(f"Mês B: {res2['period']['label']}")
                display_rain_analysis(res2["rain_df"], res2["period"]["start"], res2["period"]["end"], area_name)
        
        elif analysis_mode == "Comparativo Anual":
            st.header(f"Comparativo Anual de Precipitação para: {mes_nome}")
            display_annual_rain(analysis_results, mes_nome)

    # --- ABA 2: QUEIMADAS ---
    with tab2:
        if st.session_state.region_gdf is None:
            st.warning("Upload de shapefile/GeoJSON necessário para análise de queimadas.")
        elif not firms_api_key:
            st.error("Chave da API da FIRMS necessária.")
        else:
            if analysis_mode == "Mês Único":
                res = analysis_results[0]
                display_firms_analysis(res["firms_data"], res["period"]["start"], res["period"]["end"], st.session_state.region_gdf, area_name)
            
            elif analysis_mode == "Análise Anual Específica":
                res = analysis_results[0]
                display_single_year_firms_detail(res["firms_data"], res["period"]["label"], st.session_state.region_gdf, area_name)

            elif analysis_mode == "Comparativo Mensal":
                st.header("Comparativo de Queimadas")
                c1, c2 = st.columns(2)
                res1, res2 = analysis_results[0], analysis_results[1]
                with c1:
                    st.subheader(f"Mês A: {res1['period']['label']}")
                    display_firms_analysis(res1["firms_data"], res1["period"]["start"], res1["period"]["end"], st.session_state.region_gdf, area_name)
                with c2:
                    st.subheader(f"Mês B: {res2['period']['label']}")
                    display_firms_analysis(res2["firms_data"], res2["period"]["start"], res2["period"]["end"], st.session_state.region_gdf, area_name)
            
            elif analysis_mode == "Comparativo Anual":
                st.header(f"Comparativo Anual de Queimadas para: {mes_nome}")
                display_annual_firms(analysis_results, mes_nome)

    # --- ABA 3: COMBINADO ---
    with tab3:
        if st.session_state.region_gdf is None:
            st.warning("Upload de shapefile/GeoJSON necessário para análise combinada.")
        elif not firms_api_key:
            st.error("Chave da API da FIRMS necessária.")
        else:
            if analysis_mode in ["Mês Único", "Análise Anual Específica"]:
                res = analysis_results[0]
                display_combined_analysis(res["rain_df"], res["firms_data"], res["period"]["start"], res["period"]["end"], area_name)
            
            elif analysis_mode == "Comparativo Mensal":
                st.header("Comparativo Combinado: Chuva vs. Queimadas")
                c1, c2 = st.columns(2)
                res1, res2 = analysis_results[0], analysis_results[1]
                with c1:
                    st.subheader(f"Mês A: {res1['period']['label']}")
                    display_combined_analysis(res1["rain_df"], res1["firms_data"], res1["period"]["start"], res1["period"]["end"], area_name)
                with c2:
                    st.subheader(f"Mês B: {res2['period']['label']}")
                    display_combined_analysis(res2["rain_df"], res2["firms_data"], res2["period"]["start"], res2["period"]["end"], area_name)

            elif analysis_mode == "Comparativo Anual":
                st.header(f"Comparativo Anual Combinado para: {mes_nome}")
                display_annual_combined(analysis_results, mes_nome)

else:
    st.info("Selecione uma área, período e clique em 'Analisar' na barra lateral.")


# Rodapé
st.divider()
st.markdown("🔗 **Referência:** [NASA POWER](https://power.larc.nasa.gov/) — parâmetro `PRECTOTCORR` (Precipitação diária corrigida, mm/dia).")
