# `f1-mcp`

A Model Context Protocol (MCP) server that provides access to Formula 1 data including race results, driver information, lap times, telemetry, and circuit details using the FastF1 library.

## Installation

Installation is done using `hatch`.

```bash
pip install f1-mcp
```

## Running the Server

Using `hatch`:

```bash
hatch run f1_mcp_server.py
```

Using `python` directly:

```bash
python src/f1_mcp/f1_mcp_server.py
```

Using the MCP inspector:

```bash
npx @modelcontextprotocol/inspector python src/f1_mcp/f1_mcp_server.py
```

Make sure to set a large timeout for requests, FastF1 takes a long time to load data upon startup.

The server will start and create a local cache directory (`f1_data_cache`) to store F1 data for faster subsequent requests.

## Claude Desktop Configuration

Add the following configuration to your `claude_desktop_config.json` file:

```json
{
  "mcpServers": {
    "f1-stats": {
      "command": "python",
      "args": ["path/to/your/f1_mcp_server.py"],
      "env": {}
    }
  }
}
```

Replace `path/to/your/f1_mcp_server.py` with the actual path to your server file.

## Available Endpoints

### Driver Information
- **`get_drivers_tool`** - Get F1 drivers for a season, optionally filtered by name or code query

### Race Results
- **`get_race_results_tool`** - Get race results for a season, optionally filtered by specific race name
- **`get_session_results_tool`** - Get session results for a specific race and session type (FP1, FP2, FP3, Qualifying, Sprint, Race)

### Circuit Information
- **`get_circuit_info_tool`** - Get circuit information and event details for a specific race

### Lap Data
- **`get_driver_laps_tool`** - Get all lap data for a specific driver in a specific session
- **`get_fastest_lap_tool`** - Get the fastest lap information for a specific session

### Telemetry
- **`get_lap_telemetry_tool`** - Get detailed telemetry data (speed, throttle, brake, etc.) for a specific lap

## Notes

- Make sure to set a large timeout for requests, FastF1 takes a long time to load data upon startup.
- The server automatically caches F1 data locally to improve performance
- First-time requests for a season may take longer as data is downloaded and cached
- Session types include: FP1, FP2, FP3, Q1, Q2, Q3, Sprint, Race
- Driver codes are typically 3-letter abbreviations (e.g., HAM, VER, LEC)