# environment/weather_patterns.py V4

"""
Weather patterns and seasonal variations for the prosumer simulation.
"""
import numpy as np
from ..utils.parameters import *

class WeatherPatterns:
    """
    Manages weather patterns and seasonal variations.
    """
    def __init__(self):
        """
        Initialize the weather patterns manager.
        """
        # Monthly distribution of day types (from Table 1 in manuscript)
        self.monthly_days = {
            1: {"sunny": {"weekday": 4, "weekend": 1}, "mixed": {"weekday": 8, "weekend": 3}, "cloudy": {"weekday": 11, "weekend": 4}},  # January
            2: {"sunny": {"weekday": 4, "weekend": 2}, "mixed": {"weekday": 7, "weekend": 3}, "cloudy": {"weekday": 9, "weekend": 3}},   # February
            3: {"sunny": {"weekday": 6, "weekend": 3}, "mixed": {"weekday": 9, "weekend": 3}, "cloudy": {"weekday": 7, "weekend": 3}},   # March
            4: {"sunny": {"weekday": 9, "weekend": 3}, "mixed": {"weekday": 9, "weekend": 3}, "cloudy": {"weekday": 4, "weekend": 2}},   # April
            5: {"sunny": {"weekday": 11, "weekend": 5}, "mixed": {"weekday": 9, "weekend": 3}, "cloudy": {"weekday": 2, "weekend": 1}},  # May
            6: {"sunny": {"weekday": 12, "weekend": 5}, "mixed": {"weekday": 8, "weekend": 3}, "cloudy": {"weekday": 1, "weekend": 1}},  # June
            7: {"sunny": {"weekday": 14, "weekend": 5}, "mixed": {"weekday": 6, "weekend": 3}, "cloudy": {"weekday": 2, "weekend": 1}},  # July
            8: {"sunny": {"weekday": 14, "weekend": 6}, "mixed": {"weekday": 6, "weekend": 2}, "cloudy": {"weekday": 2, "weekend": 1}},  # August
            9: {"sunny": {"weekday": 10, "weekend": 4}, "mixed": {"weekday": 9, "weekend": 3}, "cloudy": {"weekday": 3, "weekend": 1}},  # September
            10: {"sunny": {"weekday": 8, "weekend": 3}, "mixed": {"weekday": 9, "weekend": 3}, "cloudy": {"weekday": 6, "weekend": 2}},  # October
            11: {"sunny": {"weekday": 6, "weekend": 2}, "mixed": {"weekday": 9, "weekend": 3}, "cloudy": {"weekday": 7, "weekend": 3}},  # November
            12: {"sunny": {"weekday": 4, "weekend": 1}, "mixed": {"weekday": 8, "weekend": 3}, "cloudy": {"weekday": 11, "weekend": 4}}   # December
        }
        
        # Monthly average temperatures (from Table 1)
        self.monthly_temps = {
            1: 5.0,   # January
            2: 6.3,   # February
            3: 10.0,  # March
            4: 15.0,  # April
            5: 20.0,  # May
            6: 23.7,  # June
            7: 25.0,  # July
            8: 23.7,  # August
            9: 20.0,  # September
            10: 15.0, # October
            11: 10.0, # November
            12: 6.3   # December
        }
        
        # Day type variation factors (from Table 4)
        self.day_variations = DAY_TYPE_VARIATIONS
    
    def get_representative_days(self, month):
        """
        Get the representative days for a specific month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            list: List of tuples (day_type, count, weather_type, day_of_week)
        """
        rep_days = []
        
        for weather in ["sunny", "mixed", "cloudy"]:
            for day_type in ["weekday", "weekend"]:
                count = self.monthly_days[month][weather][day_type]
                if count > 0:
                    rep_days.append((
                        f"{weather}_{day_type}",  # Day type identifier
                        count,                    # Number of this day type in month
                        weather,                  # Weather type
                        day_type                  # Day of week type
                    ))
        
        return rep_days
    
    def get_consumption_multiplier(self, month, day_type):
        """
        Calculate the consumption multiplier for a specific month and day type.
        
        Args:
            month: Month number (1-12)
            day_type: Day type identifier (e.g., 'sunny_weekday')
            
        Returns:
            float: Combined consumption multiplier
        """
        # Temperature-dependent seasonal factor
        temp = self.monthly_temps[month]
        season_factor = 1 + (
            HEATING_SENSITIVITY * max(0, REFERENCE_HEATING_TEMP - temp) +
            COOLING_SENSITIVITY * max(0, temp - COMFORT_COOLING_TEMP)
        )
        
        # Day type factor
        day_factor = self.day_variations[day_type]["consumption"]
        
        return season_factor * day_factor
    
    def get_generation_multiplier(self, month, day_type):
        """
        Calculate the generation multiplier for a specific month and day type.
        
        Args:
            month: Month number (1-12)
            day_type: Day type identifier (e.g., 'sunny_weekday')
            
        Returns:
            float: Generation multiplier
        """
        # Seasonal factor for solar generation (based on average monthly insolation)
        # Values normalized where summer (July) = 1.0, winter (January) = 0.3
        seasonal_factors = {
            1: 0.3,  # January
            2: 0.4,  # February
            3: 0.5,  # March
            4: 0.7,  # April
            5: 0.85, # May
            6: 0.95, # June
            7: 1.0,  # July
            8: 0.95, # August
            9: 0.8,  # September
            10: 0.6, # October
            11: 0.4, # November
            12: 0.3  # December
        }
        
        # Day type factor
        day_factor = self.day_variations[day_type]["generation"]
        
        return seasonal_factors[month] * day_factor
    
    def get_total_days_in_month(self, month):
        """
        Get the total number of days in a month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            int: Total days in the month
        """
        days_in_month = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        return days_in_month[month]