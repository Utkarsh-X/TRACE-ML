/**
 * Location Autocomplete Component
 *
 * Provides country dropdown and city autocomplete for person enrollment.
 */

(function() {
  "use strict";

  window.LocationAutocomplete = {
    _apiBase: "http://127.0.0.1:8080",  // Service URL
    _currentCountry: "US",
    _citySuggestions: [],

    init: function() {
      this.setupElements();
      this.loadGeoData();
      this.attachEventListeners();
    },

    setupElements: function() {
      var form = document.querySelector("form, [role='form']");
      if (!form) return;

      // Find or create location fields section
      var geoSection = document.querySelector(".geo-section");
      if (geoSection) {
        // Already exists, just enhance
        this.enhanceExistingFields();
      } else {
        // Create new section
        this.createLocationFields();
      }
    },

    createLocationFields: function() {
      // This would be added to enrollment form
      // For now, we'll enhance existing fields if they exist
      this.enhanceExistingFields();
    },

    enhanceExistingFields: function() {
      // Look for city/country inputs and enhance them
      var cityInput = document.querySelector("input[placeholder*='city' i], input[placeholder*='location' i], #city, #location");
      var countryInput = document.querySelector("input[placeholder*='country' i], #country");

      if (countryInput) {
        // Replace text input with select dropdown
        this.convertToCountryDropdown(countryInput);
      }

      if (cityInput) {
        // Add autocomplete to city input
        this.addCityAutocomplete(cityInput);
      }
    },

    convertToCountryDropdown: function(existingInput) {
      var self = this;
      var parent = existingInput.parentNode;

      // Create select element
      var select = document.createElement("select");
      select.id = existingInput.id || "country-select";
      select.className = "enroll-input";
      select.style.cssText = existingInput.getAttribute("style") || "";

      select.innerHTML = '<option value="">-- Select Country --</option>';

      // Load countries from API
      fetch(self._apiBase + "/api/v1/geo/countries")
        .then(r => r.json())
        .then(countries => {
          countries.forEach(function(c) {
            var opt = document.createElement("option");
            opt.value = c.code;
            opt.textContent = c.flag + " " + c.name;
            select.appendChild(opt);
          });
        })
        .catch(function() {
          // Fallback: use embedded list
          self.loadCountriesFallback(select);
        });

      select.addEventListener("change", function() {
        self._currentCountry = this.value;
        self.loadCitySuggestions(self._currentCountry);
      });

      // Replace existing input
      parent.replaceChild(select, existingInput);
      existingInput._select = select;
    },

    loadCountriesFallback: function(select) {
      // Embedded country list for fallback
      var countries = [
        { code: "US", name: "United States", flag: "🇺🇸" },
        { code: "GB", name: "United Kingdom", flag: "🇬🇧" },
        { code: "CA", name: "Canada", flag: "🇨🇦" },
        { code: "AU", name: "Australia", flag: "🇦🇺" },
        { code: "IN", name: "India", flag: "🇮🇳" },
        { code: "JP", name: "Japan", flag: "🇯🇵" },
      ];

      countries.forEach(function(c) {
        var opt = document.createElement("option");
        opt.value = c.code;
        opt.textContent = c.flag + " " + c.name;
        select.appendChild(opt);
      });
    },

    addCityAutocomplete: function(input) {
      var self = this;

      // Create suggestions container
      var suggestionsDiv = document.createElement("div");
      suggestionsDiv.className = "city-suggestions";
      suggestionsDiv.style.cssText = `
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: #2a2a2a;
        border: 1px solid rgba(255,255,255,0.2);
        border-top: none;
        border-radius: 0 0 4px 4px;
        max-height: 200px;
        overflow-y: auto;
        z-index: 10;
        display: none;
      `;
      input.parentNode.style.position = "relative";
      input.parentNode.appendChild(suggestionsDiv);

      input.addEventListener("input", function(e) {
        var query = e.target.value;

        if (query.length < 2) {
          suggestionsDiv.style.display = "none";
          return;
        }

        // Fetch matching cities
        var url = self._apiBase + "/api/v1/geo/cities/search?country=" + self._currentCountry + "&query=" + encodeURIComponent(query);
        fetch(url)
          .then(r => r.json())
          .then(function(suggestions) {
            if (suggestions.length === 0) {
              suggestionsDiv.style.display = "none";
              return;
            }

            suggestionsDiv.innerHTML = suggestions.map(function(city) {
              return `<div class="city-suggestion" data-city="${city.name}" style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 0.75rem;">
                ${city.name}
              </div>`;
            }).join("");

            suggestionsDiv.style.display = "block";

            // Attach click handlers
            suggestionsDiv.querySelectorAll(".city-suggestion").forEach(function(el) {
              el.addEventListener("click", function() {
                input.value = this.getAttribute("data-city");
                suggestionsDiv.style.display = "none";
              });
            });
          })
          .catch(function() {
            suggestionsDiv.style.display = "none";
          });
      });

      // Hide suggestions when clicking outside
      document.addEventListener("click", function(e) {
        if (e.target !== input && !suggestionsDiv.contains(e.target)) {
          suggestionsDiv.style.display = "none";
        }
      });
    },

    loadGeoData: function() {
      // Load data from embedded geo.json or API
      var self = this;
      fetch("/shared/geo.json")
        .then(r => r.json())
        .then(function(data) {
          self._geoData = data;
        })
        .catch(function() {
          console.log("Geo data not available, using API fallback");
        });
    },

    loadCitySuggestions: function(countryCode) {
      var self = this;
      var url = self._apiBase + "/api/v1/geo/cities?country=" + countryCode;

      fetch(url)
        .then(r => r.json())
        .then(function(cities) {
          self._citySuggestions = cities;
        });
    },

    validateLocation: function(country, city, callback) {
      var self = this;
      var url = self._apiBase + "/api/v1/geo/validate?country=" + country + "&city=" + encodeURIComponent(city);

      fetch(url)
        .then(r => r.json())
        .then(callback)
        .catch(function() {
          callback({ valid: false, error: "Validation failed" });
        });
    },
  };

  // Initialize when DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function() {
      LocationAutocomplete.init();
    });
  } else {
    LocationAutocomplete.init();
  }

})();
