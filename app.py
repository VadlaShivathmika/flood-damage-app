# -----------------------------
# STEP 6: Interactive Streamlit App (Option 2)
# -----------------------------
import streamlit as st
import ee, geemap
import pandas as pd
import datetime

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

st.set_page_config(page_title="Flood Damage Assessment", layout="wide")

st.title("🌊 Flood Damage Assessment Tool")
st.write("Analyze flood impact on infrastructure using Sentinel-1 + OSM data.")

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.header("User Input")

# Location input (lat/lon)
lat = st.sidebar.number_input("Latitude", value=27.5, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=85.3, format="%.4f")

# Date input
pre_date = st.sidebar.date_input("Pre-flood Date", datetime.date(2023, 7, 1))
post_date = st.sidebar.date_input("Post-flood Date", datetime.date(2023, 7, 15))

# AOI buffer in km
buffer_km = st.sidebar.slider("AOI Buffer (km)", 5, 50, 10)

# AOI definition
aoi = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)

# -----------------------------
# Step 3: Flood Prediction (simplified threshold)
# -----------------------------
def get_sentinel1_image(date):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(date, date.advance(1, 'month'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .select('VV')
            .mean())

pre_img = get_sentinel1_image(ee.Date(str(pre_date)))
post_img = get_sentinel1_image(ee.Date(str(post_date)))

# Flood detection (difference threshold)
vv_diff = pre_img.subtract(post_img)
flood_mask = vv_diff.lt(-1.5).selfMask().rename("flooded")

# -----------------------------
# Step 4–5: Asset Detection + Loss Estimation
# -----------------------------
# Buildings from GHSL
ghsl = ee.Image("JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1").clip(aoi).select(0).gt(0).selfMask()
buildings = ghsl.reduceToVectors(
    geometry=aoi, scale=100, geometryType='polygon',
    eightConnected=True, bestEffort=True, maxPixels=1e13
)

# Mock Hospitals (real OSM fetching skipped here for demo)
hospitals = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(aoi).limit(10)

# Roads (TIGER dataset for demo)
roads = ee.FeatureCollection("TIGER/2016/Roads").filterBounds(aoi).limit(50)

# Function to mark flooded assets
def mark_flooded_assets(fc, flood_mask, name, unit_cost):
    if fc.size().getInfo() == 0:
        return fc, 0, pd.DataFrame()

    flooded = fc.map(
        lambda f: f.set('flooded',
            flood_mask.reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=f.geometry(),
                scale=100,
                maxPixels=1e13
            ).get('flooded'))
    )
    flooded_only = flooded.filter(ee.Filter.eq('flooded', 1))
    count = flooded_only.size().getInfo()

    # Convert to table with estimated costs
    coords = flooded_only.map(
        lambda f: f.set("lon", f.geometry().centroid().coordinates().get(0))
                  .set("lat", f.geometry().centroid().coordinates().get(1))
                  .set("loss_inr", unit_cost)
    ).getInfo()

    rows = []
    for f in coords['features']:
        rows.append({
            "Type": name,
            "Latitude": f['properties']['lat'],
            "Longitude": f['properties']['lon'],
            "Estimated_Loss_INR": f['properties']['loss_inr']
        })

    return flooded_only, count, pd.DataFrame(rows)

# Cost assumptions
building_cost = 500000
road_cost = 1000000
hospital_cost = 5000000

flooded_buildings, b_count, b_df = mark_flooded_assets(buildings, flood_mask, "Building", building_cost)
flooded_roads, r_count, r_df = mark_flooded_assets(roads, flood_mask, "Road", road_cost)
flooded_hospitals, h_count, h_df = mark_flooded_assets(hospitals, flood_mask, "Hospital", hospital_cost)

# Merge all results
all_df = pd.concat([b_df, r_df, h_df], ignore_index=True)
total_loss = all_df["Estimated_Loss_INR"].sum() if not all_df.empty else 0

# -----------------------------
# Summary Panel
# -----------------------------
st.subheader("📊 Flood Impact Summary")

# Compute flooded area safely
flood_area = flood_mask.multiply(ee.Image.pixelArea()) \
                       .reduceRegion(
                           reducer=ee.Reducer.sum(),
                           geometry=aoi,
                           scale=100,
                           maxPixels=1e13
                       ).get('flooded')

if flood_area is not None:
    flood_area = flood_area.getInfo() / 1e6
else:
    flood_area = 0

st.write(f"**Flooded Area:** {flood_area:.2f} sq.km")

st.metric("Flooded Buildings", b_count)
st.metric("Flooded Roads", r_count)
st.metric("Flooded Hospitals", h_count)
st.metric("💰 Estimated Total Loss (INR)", f"{total_loss:,}")

# Show detailed table
st.subheader("📑 Detailed Affected Assets")
st.dataframe(all_df)

# Download option
if not all_df.empty:
    csv = all_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Report (CSV)", data=csv,
                       file_name="Flood_Damage_Report.csv", mime="text/csv")

# -----------------------------
# Map Visualization
# -----------------------------
Map = geemap.Map(plugin_Draw=True, plugin_LayerControl=True)
Map.centerObject(aoi, 9)
Map.addLayer(flood_mask, {"palette": ["white", "blue"], "min": 0, "max": 1}, "Flood Mask")
Map.addLayer(flooded_buildings, {"color": "red"}, "Flooded Buildings")
Map.addLayer(flooded_roads, {"color": "yellow"}, "Flooded Roads")
Map.addLayer(flooded_hospitals, {"color": "purple"}, "Flooded Hospitals")

st.subheader("🗺️ Interactive Map")
Map.to_streamlit(height=600)

