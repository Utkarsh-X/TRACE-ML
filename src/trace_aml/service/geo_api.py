"""Geolocation API for location validation and autocomplete."""

from typing import Any


# ISO 3166-1 Countries (complete list)
COUNTRIES = [
    {"code": "US", "name": "United States", "flag": "🇺🇸"},
    {"code": "GB", "name": "United Kingdom", "flag": "🇬🇧"},
    {"code": "CA", "name": "Canada", "flag": "🇨🇦"},
    {"code": "AU", "name": "Australia", "flag": "🇦🇺"},
    {"code": "IN", "name": "India", "flag": "🇮🇳"},
    {"code": "JP", "name": "Japan", "flag": "🇯🇵"},
    {"code": "DE", "name": "Germany", "flag": "🇩🇪"},
    {"code": "FR", "name": "France", "flag": "🇫🇷"},
    {"code": "IT", "name": "Italy", "flag": "🇮🇹"},
    {"code": "ES", "name": "Spain", "flag": "🇪🇸"},
    {"code": "MX", "name": "Mexico", "flag": "🇲🇽"},
    {"code": "BR", "name": "Brazil", "flag": "🇧🇷"},
    {"code": "ZA", "name": "South Africa", "flag": "🇿🇦"},
    {"code": "SG", "name": "Singapore", "flag": "🇸🇬"},
    {"code": "NZ", "name": "New Zealand", "flag": "🇳🇿"},
    {"code": "NL", "name": "Netherlands", "flag": "🇳🇱"},
    {"code": "CH", "name": "Switzerland", "flag": "🇨🇭"},
    {"code": "SE", "name": "Sweden", "flag": "🇸🇪"},
    {"code": "NO", "name": "Norway", "flag": "🇳🇴"},
    {"code": "DK", "name": "Denmark", "flag": "🇩🇰"},
    {"code": "IE", "name": "Ireland", "flag": "🇮🇪"},
    {"code": "NZ", "name": "New Zealand", "flag": "🇳🇿"},
    {"code": "SG", "name": "Singapore", "flag": "🇸🇬"},
    {"code": "HK", "name": "Hong Kong", "flag": "🇭🇰"},
    {"code": "AE", "name": "United Arab Emirates", "flag": "🇦🇪"},
    {"code": "ZZ", "name": "Other/Unknown", "flag": "🌍"},
]

# Major cities by country
CITIES_BY_COUNTRY = {
    "US": [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
        {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
        {"name": "Phoenix", "lat": 33.4484, "lon": -112.0742},
        {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
        {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936},
        {"name": "San Diego", "lat": 32.7157, "lon": -117.1611},
        {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
        {"name": "San Jose", "lat": 37.3382, "lon": -121.8863},
        {"name": "Austin", "lat": 30.2672, "lon": -97.7431},
        {"name": "Jacksonville", "lat": 30.3322, "lon": -81.6557},
        {"name": "Fort Worth", "lat": 32.7555, "lon": -97.3308},
        {"name": "Columbus", "lat": 39.9612, "lon": -82.9988},
        {"name": "Indianapolis", "lat": 39.7684, "lon": -86.1581},
    ],
    "GB": [
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Manchester", "lat": 53.4808, "lon": -2.2426},
        {"name": "Birmingham", "lat": 52.5073, "lon": -1.8898},
        {"name": "Leeds", "lat": 53.8008, "lon": -1.5491},
        {"name": "Glasgow", "lat": 55.8642, "lon": -4.2518},
        {"name": "Liverpool", "lat": 53.4084, "lon": -2.9916},
        {"name": "Newcastle", "lat": 54.9783, "lon": -1.6178},
        {"name": "Sheffield", "lat": 53.3811, "lon": -1.4701},
        {"name": "Bristol", "lat": 51.4545, "lon": -2.5879},
        {"name": "Leicester", "lat": 52.6369, "lon": -1.1398},
    ],
    "CA": [
        {"name": "Toronto", "lat": 43.6629, "lon": -79.3957},
        {"name": "Montreal", "lat": 45.5017, "lon": -73.5673},
        {"name": "Vancouver", "lat": 49.2827, "lon": -123.1207},
        {"name": "Calgary", "lat": 51.0504, "lon": -114.0853},
        {"name": "Edmonton", "lat": 53.5461, "lon": -113.4938},
        {"name": "Ottawa", "lat": 45.4215, "lon": -75.6972},
        {"name": "Winnipeg", "lat": 49.8951, "lon": -97.1384},
        {"name": "Quebec City", "lat": 46.8139, "lon": -71.2080},
    ],
    "AU": [
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
        {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631},
        {"name": "Brisbane", "lat": -27.4698, "lon": 153.0251},
        {"name": "Perth", "lat": -31.9505, "lon": 115.8605},
        {"name": "Adelaide", "lat": -34.9285, "lon": 138.6007},
        {"name": "Gold Coast", "lat": -28.0028, "lon": 153.4314},
        {"name": "Newcastle", "lat": -32.9267, "lon": 151.7793},
    ],
    "IN": [
        {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        {"name": "Delhi", "lat": 28.7041, "lon": 77.1025},
        {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
        {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
        {"name": "Pune", "lat": 18.5204, "lon": 73.8567},
        {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714},
    ],
    "JP": [
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Yokohama", "lat": 35.4437, "lon": 139.6380},
        {"name": "Osaka", "lat": 34.6937, "lon": 135.5023},
        {"name": "Kyoto", "lat": 35.0116, "lon": 135.7681},
        {"name": "Kobe", "lat": 34.6901, "lon": 135.1955},
        {"name": "Sapporo", "lat": 43.0642, "lon": 141.3469},
        {"name": "Fukuoka", "lat": 33.5904, "lon": 130.4017},
    ],
}


def get_countries() -> list[dict[str, Any]]:
    """Return list of all countries."""
    return sorted(COUNTRIES, key=lambda c: c["name"])


def get_cities(country_code: str) -> list[dict[str, Any]]:
    """Get cities for a country."""
    cities = CITIES_BY_COUNTRY.get(country_code, [])
    return sorted(cities, key=lambda c: c["name"])


def search_cities(country_code: str, query: str) -> list[dict[str, Any]]:
    """Autocomplete cities by query."""
    if not query or len(query) < 2:
        return []
    
    query_lower = query.lower()
    cities = CITIES_BY_COUNTRY.get(country_code, [])
    matches = [c for c in cities if query_lower in c["name"].lower()]
    
    # Sort by relevance (starts with query first, then contains)
    return sorted(
        matches,
        key=lambda c: (
            not c["name"].lower().startswith(query_lower),
            c["name"]
        )
    )


def validate_city(country_code: str, city_name: str) -> bool:
    """Check if city exists in country."""
    cities = CITIES_BY_COUNTRY.get(country_code, [])
    return any(c["name"].lower() == city_name.lower() for c in cities)


def validate_country(country_code: str) -> bool:
    """Check if country code is valid."""
    return any(c["code"] == country_code for c in COUNTRIES)


def create_geo_router(settings: "Settings", store: "VectorStore") -> Any:
    """Create FastAPI router for geolocation endpoints."""
    try:
        from fastapi import APIRouter
    except ImportError as exc:
        raise RuntimeError("Geo API needs FastAPI: pip install fastapi") from exc

    router = APIRouter(prefix="/api/v1", tags=["geo"])

    @router.get("/geo/countries")
    def list_countries() -> list[dict[str, Any]]:
        """List all countries for dropdown."""
        return get_countries()

    @router.get("/geo/cities")
    def list_cities(country: str = "US") -> list[dict[str, Any]]:
        """Get cities for a country."""
        if not validate_country(country):
            return []
        return get_cities(country)

    @router.get("/geo/cities/search")
    def search_city_autocomplete(country: str = "US", query: str = "") -> list[dict[str, Any]]:
        """Autocomplete cities by query."""
        if not validate_country(country):
            return []
        return search_cities(country, query)

    @router.post("/geo/validate")
    def validate_location(country: str, city: str) -> dict[str, bool]:
        """Validate that city exists in country."""
        country_valid = validate_country(country)
        city_valid = validate_city(country, city) if country_valid else False
        return {
            "country_valid": country_valid,
            "city_valid": city_valid,
            "valid": country_valid and city_valid,
        }

    return router
