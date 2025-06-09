import asyncio
from typing import Optional
import fastf1
import fastf1.events
import pandas as pd
import functools
import fastmcp
import json
import numpy as np

# Global cache variables
seasons_cache = None
drivers_cache = {}

# Configure FastF1 to suppress warnings and enable caching
fastf1.Cache.enable_cache('f1_data_cache')  # Creates a local cache directory


def clean_for_json(obj):
    """
    Recursively clean an object to make it JSON serializable.
    Handles NaN, infinity, and other problematic values.
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj) or (isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj))):
        return None
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    else:
        return obj

async def _run_in_executor(func, *args, **kwargs):
    """
    Run a potentially blocking function in a thread executor
    to prevent blocking the event loop
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

async def get_drivers(season=None, query=None):
    """
    Retrieve F1 drivers for a specific season.
    
    :param season: Optional season to filter drivers
    :param query: Optional name/code to filter drivers
    :return: List of drivers
    """
    global drivers_cache
    if season not in drivers_cache:
        try:
            # Fetch drivers for the specific season
            events: fastf1.events.EventSchedule = await _run_in_executor(fastf1.get_event_schedule, season)
            
            # Collect unique drivers for the season
            drivers = []
            for event in events.itertuples():
                try:
                    weekend = await _run_in_executor(
                        fastf1.get_event,
                        season, 
                        event.RoundNumber,
                    )
                    
                    session = await _run_in_executor(
                        weekend.get_session, 
                        'Qualifying'  # Use Qualifying to get most drivers
                    )
                    await _run_in_executor(session.load)
                    
                    for driver in session.results.itertuples():
                        driver_info = {
                            "code": driver.DriverNumber,
                            "name": driver.FullName,
                            "team": driver.TeamName
                        }
                        if driver_info not in drivers:
                            drivers.append(driver_info)

                except Exception as e:
                    print(f"Error fetching drivers for event {event.Location}: {e}")
            
            drivers_cache[season] = drivers
        except Exception as e:
            print(f"Error fetching drivers for season {season}: {e}")
            return []
    
    filtered_drivers = drivers_cache[season]
    
    if query:
        filtered_drivers = [
            d for d in filtered_drivers 
            if query.lower() in d['name'].lower() 
            or query.lower() in d['code'].lower()
        ]
    
    return filtered_drivers

async def get_race_results(season, race_name=None):
    """
    Retrieve race results for a given season.
    
    :param season: Season to retrieve results for
    :param race_name: Optional specific race to filter
    :return: Race results
    """
    try:
        # Fetch event schedule for the season
        events = await _run_in_executor(fastf1.get_event_schedule, season)
        
        race_results = []
        for event in events.itertuples():
            try:
                # Get the race weekend
                weekend = await _run_in_executor(
                    fastf1.get_event, 
                    season,
                    event.RoundNumber
                )
                
                # Get race session
                race_session = await _run_in_executor(
                    weekend.get_session, 
                    'Race'
                )
                
                # Load session data
                await _run_in_executor(race_session.load)
                
                # Get results - this is a DataFrame property, not a function
                session_results = race_session.results
                
                # Prepare race result
                race_info = {
                    "race": event.EventName,
                    "location": event.Location,
                    "date": str(event.EventDate),
                    "winner": session_results.iloc[0]['FullName'] if len(session_results) > 0 else None,
                    "podium": list(session_results.iloc[:3]['FullName']) if len(session_results) >= 3 else [],
                    "full_results": clean_for_json(session_results.to_dict(orient='records'))
                }
                
                # Filter by race name if provided
                if not race_name or race_name.lower() in race_info['race'].lower():
                    race_results.append(race_info)
            
            except Exception as e:
                print(f"Error fetching results for {event.EventName}: {e}")
        
        return race_results
    
    except Exception as e:
        print(f"Error fetching race results for season {season}: {e}")
        return []

async def get_session_results(season, race_name, session_type):
    """
    Retrieve session results for a specific race and session.
    
    :param season: Season year
    :param race_name: Name of the race
    :param session_type: Type of session (FP1, FP2, FP3, Q1, Q2, Q3, Sprint, Race)
    :return: Session results
    """
    try:
        # Get the race weekend
        weekend = await _run_in_executor(fastf1.get_event, season, race_name)
        
        # Get the specific session
        session = await _run_in_executor(weekend.get_session, session_type)
        
        # Load session data
        await _run_in_executor(session.load)
        
        # Get session results - this is a DataFrame property, not a function
        results = session.results
        
        return {
            "race": race_name,
            "season": season,
            "session": session_type,
            "results": clean_for_json(results.to_dict(orient='records')) if results is not None else []
        }
        
    except Exception as e:
        print(f"Error fetching session results for {race_name} {session_type}: {e}")
        return {}

async def get_circuit_info(season, race_name):
    """
    Retrieve circuit information for a specific race.
    
    :param season: Season year
    :param race_name: Name of the race
    :return: Circuit information
    """
    try:
        # Get the race weekend
        weekend = await _run_in_executor(fastf1.get_event, season, race_name)
        
        # Get a session to access circuit info (using Practice 1 as default)
        session = await _run_in_executor(weekend.get_session, 'FP1')
        
        # Load session data
        await _run_in_executor(session.load)
        
        return {
            "race": race_name,
            "season": season,
            "event_name": session.event['EventName'],
            "location": session.event['Location'],
            "country": session.event['Country'],
            "date": str(session.event['EventDate'])
        }
        
    except Exception as e:
        print(f"Error fetching circuit info for {race_name}: {e}")
        return {}

async def get_driver_laps(season, race_name, session_type, driver_code):
    """
    Retrieve all laps for a specific driver in a specific session.
    
    :param season: Season year
    :param race_name: Name of the race
    :param session_type: Type of session
    :param driver_code: Driver code (e.g., 'HAM', 'VER')
    :return: All laps for the driver
    """
    try:
        # Get the race weekend
        weekend = await _run_in_executor(fastf1.get_event, season, race_name)
        
        # Get the specific session
        session = await _run_in_executor(weekend.get_session, session_type)
        
        # Load session data
        await _run_in_executor(session.load)
        
        # Get laps for the specific driver
        driver_laps = session.laps.pick_driver(driver_code)
        
        return {
            "race": race_name,
            "season": season,
            "session": session_type,
            "driver": driver_code,
            "laps": clean_for_json(driver_laps.to_dict(orient='records')) if driver_laps is not None else []
        }
        
    except Exception as e:
        print(f"Error fetching laps for driver {driver_code} in {race_name} {session_type}: {e}")
        return {}

async def get_fastest_lap(season, race_name, session_type):
    """
    Retrieve the fastest lap in a session.
    
    :param season: Season year
    :param race_name: Name of the race
    :param session_type: Type of session
    :return: Fastest lap information
    """
    try:
        # Get the race weekend
        weekend = await _run_in_executor(fastf1.get_event, season, race_name)
        
        # Get the specific session
        session = await _run_in_executor(weekend.get_session, session_type)
        
        # Load session data
        await _run_in_executor(session.load)
        
        # Get the fastest lap
        fastest_lap = session.laps.pick_fastest()
        
        return {
            "race": race_name,
            "season": season,
            "session": session_type,
            "fastest_lap": clean_for_json(fastest_lap.to_dict()) if fastest_lap is not None else {}
        }
        
    except Exception as e:
        print(f"Error fetching fastest lap for {race_name} {session_type}: {e}")
        return {}

async def get_lap_telemetry(season, race_name, session_type, driver_code, lap_number):
    """
    Retrieve telemetry data for a specific lap.
    
    :param season: Season year
    :param race_name: Name of the race
    :param session_type: Type of session
    :param driver_code: Driver code
    :param lap_number: Lap number
    :return: Telemetry data for the lap
    """
    try:
        # Get the race weekend
        weekend = await _run_in_executor(fastf1.get_event, season, race_name)
        
        # Get the specific session
        session = await _run_in_executor(weekend.get_session, session_type)
        
        # Load session data
        await _run_in_executor(session.load)
        
        # Get the specific lap
        lap = session.laps.pick_driver(driver_code).pick_lap(lap_number)
        
        # Get telemetry for the lap
        telemetry = await _run_in_executor(lap.get_telemetry)
        
        return {
            "race": race_name,
            "season": season,
            "session": session_type,
            "driver": driver_code,
            "lap_number": lap_number,
            "telemetry": clean_for_json(telemetry.to_dict(orient='records')) if telemetry is not None else []
        }
        
    except Exception as e:
        print(f"Error fetching telemetry for lap {lap_number} of driver {driver_code}: {e}")
        return {}

async def main():
    # Create the FastMCP server
    server = fastmcp.FastMCP()

    @server.tool()
    async def get_drivers_tool(season: Optional[int] = None, query: Optional[str] = None) -> list:
        """Get F1 drivers for a season, optionally filtered by query"""
        return await get_drivers(season, query)

    @server.tool()
    async def get_race_results_tool(season: int, race_name: Optional[str] = None) -> list:
        """Get race results for a season, optionally filtered by race name"""
        return await get_race_results(season, race_name)

    @server.tool()
    async def get_session_results_tool(season: int, race_name: str, session_type: str) -> list:
        """Get session results for a specific race and session"""
        return await get_session_results(season, race_name, session_type)

    @server.tool()
    async def get_circuit_info_tool(season: int, race_name: str) -> list:
        """Get circuit information for a specific race"""
        return await get_circuit_info(season, race_name)

    @server.tool()
    async def get_driver_laps_tool(season: int, race_name: str, session_type: str, driver_code: str) -> list:
        """Get all laps for a specific driver in a specific session"""
        return await get_driver_laps(season, race_name, session_type, driver_code)

    @server.tool()
    async def get_fastest_lap_tool(season: int, race_name: str, session_type: str) -> list:
        """Get the fastest lap in a session"""
        return await get_fastest_lap(season, race_name, session_type)

    @server.tool()
    async def get_lap_telemetry_tool(season: int, race_name: str, session_type: str, driver_code: str, lap_number: int) -> list:
        """Get telemetry data for a specific lap"""
        return await get_lap_telemetry(season, race_name, session_type, driver_code, lap_number)
    
    await server.run_async("stdio")
    print("F1 Stats MCP Server running on localhost:8000")
    
    # Keep the server running
    await asyncio.get_event_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())
    