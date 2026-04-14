/**
 * Country State City API Integration
 * Professional hierarchical location selection
 * Using: https://api.countrystatecity.in/v1
 */

(function() {
  "use strict";

  // Configuration
  var Config = {
    API_KEY: "392cea6b787ad2e7b4c3b624140bf4354a241a8c2e9868a0de53590dc65c52a6",
    API_BASE: "https://api.countrystatecity.in/v1",
    PROXY_ENDPOINT: "/api/v1/location/proxy", // Fallback if direct CORS issues
  };

  // State management
  var LocationState = {
    countries: [],
    states: [],
    cities: [],
    selectedCountry: null,
    selectedState: null,
    selectedCity: null,
  };

  // DOM elements
  var Elements = {};

  window.LocationAPI = {
    // Initialize the location component
    init: function() {
      this.createFormElements();
      this.attachEventListeners();
      this.loadCountries();
    },

    // Create form structure
    createFormElements: function() {
      var enrollForm = document.getElementById("enroll-form");
      if (!enrollForm) return;

      // Find location section or create it
      var locationSection = document.querySelector(".location-section");
      if (!locationSection) {
        locationSection = document.createElement("div");
        locationSection.className = "location-section";
        var firstField = enrollForm.querySelector(".form-group");
        if (firstField) {
          firstField.parentNode.insertBefore(locationSection, firstField);
        } else {
          enrollForm.appendChild(locationSection);
        }
      }

      locationSection.innerHTML = `
        <h3 class="text-[0.9rem] font-headline font-semibold mb-3">📍 Location</h3>
        
        <div class="form-group mb-3">
          <label for="country-select" class="block text-[0.75rem] font-semibold mb-1">
            Country *
          </label>
          <select id="country-select" class="form-input" required>
            <option value="">Loading countries...</option>
          </select>
          <div id="country-error" class="text-[0.65rem] text-red-500 mt-1" style="display: none;"></div>
        </div>

        <div class="form-group mb-3" id="state-group" style="display: none;">
          <label for="state-select" class="block text-[0.75rem] font-semibold mb-1">
            State / Province *
          </label>
          <select id="state-select" class="form-input" required>
            <option value="">Select a state...</option>
          </select>
          <div id="state-error" class="text-[0.65rem] text-red-500 mt-1" style="display: none;"></div>
        </div>

        <div class="form-group mb-3" id="city-group" style="display: none;">
          <label for="city-select" class="block text-[0.75rem] font-semibold mb-1">
            City *
          </label>
          <select id="city-select" class="form-input" required>
            <option value="">Select a city...</option>
          </select>
          <div id="city-error" class="text-[0.65rem] text-red-500 mt-1" style="display: none;"></div>
        </div>

        <div id="location-status" class="text-[0.65rem] text-outline mt-2" style="display: none;"></div>
      `;

      // Store element references
      Elements = {
        countrySelect: document.getElementById("country-select"),
        stateSelect: document.getElementById("state-select"),
        citySelect: document.getElementById("city-select"),
        stateGroup: document.getElementById("state-group"),
        cityGroup: document.getElementById("city-group"),
        countryError: document.getElementById("country-error"),
        stateError: document.getElementById("state-error"),
        cityError: document.getElementById("city-error"),
        locationStatus: document.getElementById("location-status"),
      };
    },

    // Attach event listeners
    attachEventListeners: function() {
      var self = this;

      if (Elements.countrySelect) {
        Elements.countrySelect.addEventListener("change", function() {
          self.onCountryChange(this.value);
        });
      }

      if (Elements.stateSelect) {
        Elements.stateSelect.addEventListener("change", function() {
          self.onStateChange(this.value);
        });
      }

      if (Elements.citySelect) {
        Elements.citySelect.addEventListener("change", function() {
          self.onCityChange(this.value);
        });
      }
    },

    // Load all countries
    loadCountries: async function() {
      var self = this;
      this.showStatus("Loading countries...");

      try {
        // Use backend proxy endpoint instead of external API
        var url = "/api/v1/location/countries";
        var response = await fetch(url);

        if (!response.ok) {
          throw new Error("Failed to load countries: " + response.status);
        }

        var data = await response.json();
        LocationState.countries = Array.isArray(data) ? data : [];

        // Populate select
        this.populateCountrySelect();
        this.showStatus(""); // Clear status
      } catch (error) {
        console.error("Error loading countries:", error);
        this.showError("country", "Failed to load countries. Please refresh the page.");
        this.showStatus("Error: " + error.message);
      }
    },

    // Populate country dropdown
    populateCountrySelect: function() {
      if (!Elements.countrySelect) return;

      Elements.countrySelect.innerHTML = '<option value="">Select a country...</option>';

      LocationState.countries.forEach(function(country) {
        var option = document.createElement("option");
        option.value = country.name; // Use ISO code for API
        option.textContent = "🌍 " + country.name;
        option.dataset.iso = country.iso2; // Store ISO code for state lookup
        Elements.countrySelect.appendChild(option);
      });

      // Set default (United States)
      var usOption = Array.from(Elements.countrySelect.options).find(function(opt) {
        return opt.textContent.includes("United States");
      });
      if (usOption) {
        Elements.countrySelect.value = usOption.value;
        this.onCountryChange(usOption.value);
      }
    },

    // Handle country selection change
    onCountryChange: async function(countryName) {
      var self = this;

      LocationState.selectedCountry = countryName;
      LocationState.selectedState = null;
      LocationState.selectedCity = null;

      // Reset state and city
      if (Elements.stateSelect) {
        Elements.stateSelect.innerHTML = '<option value="">Loading states...</option>';
      }
      if (Elements.citySelect) {
        Elements.citySelect.innerHTML = '<option value="">Select a city...</option>';
        Elements.cityGroup.style.display = "none";
      }

      // Clear errors
      this.clearError("state");
      this.clearError("city");

      if (!countryName) {
        Elements.stateGroup.style.display = "none";
        return;
      }

      // Get ISO code
      var selectedOption = Elements.countrySelect.querySelector(
        "option[value='" + countryName.replace(/'/g, "\\'") + "']"
      );
      var iso = selectedOption ? selectedOption.dataset.iso : null;

      if (!iso) return;

      try {
        this.showStatus("Loading states for " + countryName + "...");

        // Use backend proxy endpoint
        var url = "/api/v1/location/states?country=" + encodeURIComponent(iso);
        var response = await fetch(url);

        if (!response.ok) {
          throw new Error("Failed to load states: " + response.status);
        }

        var data = await response.json();
        LocationState.states = Array.isArray(data) ? data : [];

        if (LocationState.states.length > 0) {
          this.populateStateSelect();
          Elements.stateGroup.style.display = "block";
          this.showStatus(""); // Clear status
        } else {
          // No states, show cities directly
          Elements.stateGroup.style.display = "none";
          Elements.cityGroup.style.display = "block";
          await this.loadCitiesForCountry(iso);
        }
      } catch (error) {
        console.error("Error loading states:", error);
        this.showError("state", "Failed to load states: " + error.message);
        this.showStatus("Error: " + error.message);
      }
    },

    // Populate state dropdown
    populateStateSelect: function() {
      if (!Elements.stateSelect) return;

      Elements.stateSelect.innerHTML = '<option value="">Select a state...</option>';

      LocationState.states.forEach(function(state) {
        var option = document.createElement("option");
        option.value = state.name;
        option.textContent = state.name;
        option.dataset.iso = state.iso2;
        Elements.stateSelect.appendChild(option);
      });
    },

    // Handle state selection change
    onStateChange: async function(stateName) {
      var self = this;

      LocationState.selectedState = stateName;
      LocationState.selectedCity = null;

      // Reset city
      if (Elements.citySelect) {
        Elements.citySelect.innerHTML = '<option value="">Loading cities...</option>';
      }

      this.clearError("city");

      if (!stateName) {
        Elements.cityGroup.style.display = "none";
        return;
      }

      // Get country and state ISO codes
      var countryOption = Elements.countrySelect.querySelector(
        "option[value='" + LocationState.selectedCountry.replace(/'/g, "\\'") + "']"
      );
      var countryISO = countryOption ? countryOption.dataset.iso : null;

      var stateOption = Elements.stateSelect.querySelector(
        "option[value='" + stateName.replace(/'/g, "\\'") + "']"
      );
      var stateISO = stateOption ? stateOption.dataset.iso : null;

      if (!countryISO || !stateISO) return;

      try {
        this.showStatus("Loading cities for " + stateName + "...");

        // Use backend proxy endpoint
        var url =
          "/api/v1/location/cities?country=" +
          encodeURIComponent(countryISO) +
          "&state=" +
          encodeURIComponent(stateISO);
        var response = await fetch(url);

        if (!response.ok) {
          throw new Error("Failed to load cities: " + response.status);
        }

        var data = await response.json();
        LocationState.cities = Array.isArray(data) ? data : [];

        this.populateCitySelect();
        Elements.cityGroup.style.display = "block";
        this.showStatus(""); // Clear status
      } catch (error) {
        console.error("Error loading cities:", error);
        this.showError("city", "Failed to load cities: " + error.message);
        this.showStatus("Error: " + error.message);
      }
    },

    // Load cities directly for country (when no states)
    loadCitiesForCountry: async function(countryISO) {
      var self = this;

      try {
        this.showStatus("Loading cities...");

        // Use backend proxy endpoint
        var url = "/api/v1/location/cities?country=" + encodeURIComponent(countryISO);
        var response = await fetch(url);

        if (!response.ok) {
          throw new Error("Failed to load cities: " + response.status);
        }

        var data = await response.json();
        LocationState.cities = Array.isArray(data) ? data : [];

        this.populateCitySelect();
        this.showStatus(""); // Clear status
      } catch (error) {
        console.error("Error loading cities:", error);
        this.showError("city", "Failed to load cities: " + error.message);
        this.showStatus("Error: " + error.message);
      }
    },

    // Populate city dropdown
    populateCitySelect: function() {
      if (!Elements.citySelect) return;

      Elements.citySelect.innerHTML = '<option value="">Select a city...</option>';

      LocationState.cities.forEach(function(city) {
        var option = document.createElement("option");
        option.value = city.name;
        option.textContent = city.name;
        Elements.citySelect.appendChild(option);
      });
    },

    // Handle city selection change
    onCityChange: function(cityName) {
      LocationState.selectedCity = cityName;
      this.clearError("city");
      this.showStatus("Selected: " + LocationState.selectedCountry);
      if (LocationState.selectedState) {
        this.showStatus("Selected: " + LocationState.selectedCountry + " > " + LocationState.selectedState);
      }
      if (LocationState.selectedCity) {
        this.showStatus("Selected: " + LocationState.selectedCountry + " > " + LocationState.selectedCity);
      }
    },

    // Show error message
    showError: function(field, message) {
      var errorEl = Elements[field + "Error"];
      if (errorEl) {
        errorEl.textContent = message;
        errorEl.style.display = "block";
      }
    },

    // Clear error message
    clearError: function(field) {
      var errorEl = Elements[field + "Error"];
      if (errorEl) {
        errorEl.textContent = "";
        errorEl.style.display = "none";
      }
    },

    // Show status message
    showStatus: function(message) {
      if (Elements.locationStatus) {
        if (message) {
          Elements.locationStatus.textContent = message;
          Elements.locationStatus.style.display = "block";
        } else {
          Elements.locationStatus.style.display = "none";
        }
      }
    },

    // Get selected values
    getSelected: function() {
      return {
        country: LocationState.selectedCountry,
        state: LocationState.selectedState,
        city: LocationState.selectedCity,
      };
    },
  };

  // Auto-initialize
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function() {
      window.LocationAPI.init();
    });
  } else {
    window.LocationAPI.init();
  }

})();
