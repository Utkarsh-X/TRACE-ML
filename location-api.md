> ## Documentation Index
> Fetch the complete documentation index at: https://docs.countrystatecity.in/llms.txt
> Use this file to discover all available pages before exploring further.

# Country State City API

> Access comprehensive geographical data with our RESTful API featuring 247+ countries, 5000+ states, and 151,000+ cities

## Overview

The Country State City API provides developers with comprehensive access to geographical data covering 247+ countries, 5000+ states/provinces, and over 151,000 cities worldwide. Our RESTful API is designed for high performance, reliability, and ease of integration.

<Card title="Get your API key" icon="key" href="https://app.countrystatecity.in" horizontal>
  Register and start accessing geographical data immediately.
</Card>

## Key Features

<CardGroup cols={2}>
  <Card title="Comprehensive Coverage" icon="globe">
    Access complete geographical hierarchies from countries down to individual cities with detailed metadata.
  </Card>

  <Card title="High Performance" icon="bolt">
    Optimized endpoints with efficient data structures and caching for fast response times.
  </Card>

  <Card title="Developer Friendly" icon="code">
    RESTful design with consistent responses, comprehensive error handling, and extensive documentation.
  </Card>

  <Card title="Always Current" icon="clock">
    Regular updates ensure data accuracy with automated validation and quality checks.
  </Card>
</CardGroup>

## Quick Start

<Steps>
  <Step title="Get Your API Key">
    Register at [our portal](https://app.countrystatecity.in) to obtain your free API key.
  </Step>

  <Step title="Make Your First Request">
    Use your API key to fetch all countries:

    <CodeGroup>
      ```bash cURL theme={null}
      curl -X GET 'https://api.countrystatecity.in/v1/countries' \
        -H 'X-CSCAPI-KEY: YOUR_API_KEY'
      ```

      ```javascript JavaScript theme={null}
      const response = await fetch('https://api.countrystatecity.in/v1/countries', {
        headers: { 'X-CSCAPI-KEY': 'YOUR_API_KEY' }
      });
      const countries = await response.json();
      ```

      ```python Python theme={null}
      import requests

      response = requests.get(
        'https://api.countrystatecity.in/v1/countries',
        headers={'X-CSCAPI-KEY': 'YOUR_API_KEY'}
      )
      countries = response.json()
      ```
    </CodeGroup>
  </Step>

  <Step title="Explore the Endpoints">
    Browse our comprehensive endpoint documentation to integrate geographical data into your application.

    <Check>
      All endpoints return consistent JSON responses with proper HTTP status codes.
    </Check>
  </Step>
</Steps>

## Base URL

```
https://api.countrystatecity.in/v1
```

## Authentication

All API requests require authentication using an API key sent in the request header:

```
X-CSCAPI-KEY: YOUR_API_KEY
```

<Warning>
  Never expose your API key in client-side code or public repositories. Use environment variables or server-side configurations to keep your key secure.
</Warning>

## Available Endpoints

<CardGroup cols={3}>
  <Card title="Countries" icon="flag" href="/api/endpoints/get-all-countries">
    Get all countries or specific country details with ISO codes, currencies, and regional information.
  </Card>

  <Card title="States" icon="map" href="/api/endpoints/get-all-states">
    Access states, provinces, and regions with filtering by country and detailed geographical data.
  </Card>

  <Card title="Cities" icon="building" href="/api/endpoints/get-cities-by-country">
    Retrieve cities with coordinates, timezone data, and hierarchical filtering by country and state.
  </Card>
</CardGroup>

### Countries Endpoints

* [Get All Countries](/api/endpoints/get-all-countries) - Retrieve list of all countries
* [Get Country Details](/api/endpoints/get-country-details) - Get detailed information for a specific country

### States Endpoints

* [Get All States](/api/endpoints/get-all-states) - Retrieve list of all states
* [Get States by Country](/api/endpoints/get-states-by-country) - Get states within a specific country
* [Get State Details](/api/endpoints/get-state-details) - Get detailed information for a specific state

### Cities Endpoints

* [Get Cities by Country](/api/endpoints/get-cities-by-country) - Get cities within a specific country
* [Get Cities by State](/api/endpoints/get-cities-by-state) - Get cities within a specific state

## Response Format

All API endpoints return JSON responses with consistent structure:

### Success Response

```json  theme={null}
{
  "data": [...], // Array of results or single object
  "total": 247,  // Total count (for list endpoints)
  "page": 1      // Current page (for paginated endpoints)
}
```

### Error Response

```json  theme={null}
{
  "error": "Unauthorized. You shouldn't be here.",
  "code": 401,
  "message": "Invalid or missing API key"
}
```

## HTTP Status Codes

| Code | Description                                             |
| ---- | ------------------------------------------------------- |
| 200  | Success - Request completed successfully                |
| 401  | Unauthorized - Invalid or missing API key               |
| 404  | Not Found - Resource doesn't exist                      |
| 500  | Internal Server Error - Something went wrong on our end |

<Info>
  For 5xx errors, implement exponential backoff in your retry logic. Most issues are temporary and resolve quickly.
</Info>

## Need Help?

<CardGroup cols={2}>
  <Card title="Sample Code" icon="code" href="/api/examples">
    Complete integration examples in multiple programming languages.
  </Card>

  <Card title="Support" icon="headset" href="/api/faq">
    Common questions, troubleshooting, and community support.
  </Card>
</CardGroup>
