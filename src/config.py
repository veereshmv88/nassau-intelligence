"""
Configuration and Master Data for Nassau Intelligence:
Factory Reallocation & Shipping Optimization Recommendation System.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "nassau_candy_distributor.csv"
FALLBACK_DATA_PATH = BASE_DIR / "Nassau Candy Distributor (1).csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "nassau_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PIPELINE_PATH = MODEL_DIR / "best_lead_time_pipeline.joblib"
MODEL_METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.json"

# Factory Master Coordinates
FACTORIES = {
    "Lot's O' Nuts": {
        "lat": 32.881893,
        "lon": -111.768036,
        "city": "Casa Grande",
        "state": "Arizona",
        "region": "Pacific",
        "specialty": "Nut-based Chocolates & Crunch",
        "color": "#E9A23B"  # Warm amber
    },
    "Wicked Choccy's": {
        "lat": 32.076176,
        "lon": -81.088371,
        "city": "Savannah",
        "state": "Georgia",
        "region": "Gulf",
        "specialty": "Milk Chocolate & Caramels",
        "color": "#845EC2"  # Purple/violet
    },
    "Sugar Shack": {
        "lat": 48.119140,
        "lon": -96.181150,
        "city": "Thief River Falls",
        "state": "Minnesota",
        "region": "Interior",
        "specialty": "Hard Candies, Taffy & Powders",
        "color": "#087F5B"  # Emerald green
    },
    "Secret Factory": {
        "lat": 41.446333,
        "lon": -90.565487,
        "city": "Rock Island",
        "state": "Illinois/Iowa",
        "region": "Interior",
        "specialty": "Gumballs, Novelties & Long-life sweets",
        "color": "#2C73D2"  # Blue
    },
    "The Other Factory": {
        "lat": 35.117500,
        "lon": -89.971107,
        "city": "Memphis",
        "state": "Tennessee",
        "region": "Gulf",
        "specialty": "Toffee, Chews & Specialty Licorice",
        "color": "#D65DB1"  # Magenta/Coral
    }
}

# Initial Product Assignments
DEFAULT_PRODUCT_ASSIGNMENTS = {
    # Chocolate
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    # Sugar
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    # Other
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

PRODUCT_DIVISIONS = {
    "Wonka Bar - Nutty Crunch Surprise": "Chocolate",
    "Wonka Bar - Fudge Mallows": "Chocolate",
    "Wonka Bar -Scrumdiddlyumptious": "Chocolate",
    "Wonka Bar - Milk Chocolate": "Chocolate",
    "Wonka Bar - Triple Dazzle Caramel": "Chocolate",
    "Laffy Taffy": "Sugar",
    "SweeTARTS": "Sugar",
    "Nerds": "Sugar",
    "Fun Dip": "Sugar",
    "Everlasting Gobstopper": "Sugar",
    "Hair Toffee": "Sugar",
    "Fizzy Lifting Drinks": "Other",
    "Lickable Wallpaper": "Other",
    "Wonka Gum": "Other",
    "Kazookles": "Other"
}

# State / Province Centroid Coordinates for Great-Circle Distance
STATE_COORDINATES = {
    # United States
    "Alabama": (32.806671, -86.791130),
    "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221),
    "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564),
    "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371),
    "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817),
    "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074),
    "Hawaii": (21.094318, -157.498337),
    "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137),
    "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526),
    "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067),
    "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927),
    "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106),
    "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192),
    "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368),
    "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082),
    "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896),
    "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482),
    "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419),
    "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915),
    "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938),
    "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780),
    "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828),
    "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461),
    "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686),
    "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494),
    "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508),
    "Wyoming": (42.755966, -107.302490),
    # Canadian Provinces (in the dataset)
    "Ontario": (51.253775, -85.323214),
    "Quebec": (52.939916, -73.549136),
    "British Columbia": (53.726668, -127.647621),
    "Alberta": (53.933271, -116.576504),
    "Manitoba": (53.760860, -98.813876),
    "Saskatchewan": (52.939916, -106.450864),
    "Nova Scotia": (44.681986, -63.744311),
    "New Brunswick": (46.565316, -66.461916)
}

# Regional Centroids fallback
REGION_COORDINATES = {
    "Pacific": (38.5, -120.0),
    "Interior": (42.0, -93.0),
    "Gulf": (32.5, -86.5),
    "Atlantic": (40.5, -76.0)
}

# Region aliases / display labels
REGION_ALIASES = {
    "Pacific": "West (Pacific)",
    "Interior": "Midwest (Interior)",
    "Gulf": "South (Gulf)",
    "Atlantic": "Northeast (Atlantic)"
}

# Design System CSS Variables
THEME = {
    "bg_main": "#F5F5F0",
    "bg_secondary": "#EAF1EB",
    "primary_green": "#087F5B",
    "deep_green": "#075E46",
    "mint_accent": "#A7E8C8",
    "teal_accent": "#5AB9AD",
    "lavender_accent": "#B8B9FF",
    "coral_accent": "#FF8FAB",
    "amber_warning": "#E9A23B",
    "text_main": "#172A25",
    "text_secondary": "#64736D",
    "border_glass": "rgba(255, 255, 255, 0.72)",
    "glass_surface": "rgba(255, 255, 255, 0.58)",
    "glass_surface_strong": "rgba(255, 255, 255, 0.78)",
    "success": "#16865B",
    "error": "#C74747"
}
