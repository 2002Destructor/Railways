import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})

import json
import csv 
import math 

# --- CONFIGURATION ---

MIN_STOP_TIME = 0.5
MIN_HEADWAY = 3.0
BIG_M = 2000
DELTA_WEIGHT = 600.0
# --- PEAK WINDOW (1-hour toy example) ---
PEAK_START = 480.0   # 08:00
PEAK_END   = 540.0   # 09:00


# --- SOLVER PARAMETERS ---
NEW_TIME_LIMIT = 600
NEW_MIP_GAP = 0.001  # 0.1% relative gap
FEASIBILITY_TOL = 1e-5  # 0.00001 minutes absolute tolerance

# --- STATION AND COLUMN DEFINITIONS ---
STATIONS_ORDER = ['CCG', 'MCT', 'DDR', 'BA', 'ADH', 'GMN', 'BVI', 'BYR', 'BSR', 'VR', 'DRD']
OUTPUT_COLS_STATION_EVENTS = [
    'CCGa', 'CCGd', 'MCTa', 'MCTd', 'DDRa', 'DDRd', 'BAa', 'BAd', 'ADHa', 'ADHd',
    'GMNa', 'GMNd', 'BVIa', 'BVId', 'BYRa', 'BYRd', 'BSRa', 'BSRd', 'VRa', 'VRd',
    'DRDa', 'DRDd'
]

def load_and_validate_data():
    """Load all data files and validate consistency"""
    print("Loading data files...")
    
    # Load CSC file
    csc_df = pd.read_csv('1-o-event-ids.csv')
    print(f"Loaded CSC file: {len(csc_df)} services")
    
    # Load patterns
    patterns_df = pd.read_csv('OWR-patterns-consolidated-sequential.csv')
    print(f"Loaded patterns: {len(patterns_df)} patterns")
    
    # Load constraints
    with open('constraints_f_s.json', 'r') as f:
        constraints = json.load(f)
    print("Loaded constraints file")
    
    return csc_df, patterns_df, constraints

def extract_travel_times(patterns_df):
    """Extract travel times between stations from patterns - create bidirectional mapping"""
    travel_times = {}
    
    for _, row in patterns_df.iterrows():
        pattern_id = row['Pattern_ID']
        travel_times[pattern_id] = {}
        
        # Process all segments
        for i in range(1, 11):
            seg_col = f'Major_Segment_{i}'
            time_col = f'Time_{i}'
            
            if pd.notna(row[seg_col]) and pd.notna(row[time_col]):
                segment = str(row[seg_col]).strip()
                if '-' in segment:
                    from_st, to_st = segment.split('-')
                    travel_time = float(row[time_col])
                    
                    # Store both directions with same travel time
                    travel_times[pattern_id][f"{from_st}-{to_st}"] = travel_time
                    travel_times[pattern_id][f"{to_st}-{from_st}"] = travel_time  # Reverse direction
                    
                    print(f"  {pattern_id}: {from_st}->{to_st} = {travel_time} min (bidirectional)")
    
    return travel_times

def build_service_sequences(csc_df):
    """Build proper event sequences for each service following CSC column order"""
    services = []
    all_event_ids = set()
    
    # Get station event columns in correct order
    event_cols = [col for col in csc_df.columns if col.endswith(('a', 'd'))]
    
    for _, row in csc_df.iterrows():
        service = {
            'SrNum': int(row['SrNum']),  # Ensure integer
            'OriginalTime': float(row['Time']),  # Ensure float
            'Type': row['Type'],
            'Dir': row['Dir'],
            'PatNum': str(row['PatNum']),  # Keep as string for pattern matching
            'From': row['From'],
            'To': row['To'],
            'Events': [],
            'FirstDeparture': None,
            'LastArrival': None
        }
        
        # Build event sequence in CSC column order
        events_in_order = []
        for col in event_cols:
            if pd.notna(row[col]) and row[col] != '':
                try:
                    event_id = int(row[col])
                    station = col[:-1]
                    event_type = 'arrival' if col.endswith('a') else 'departure'
                    
                    event_data = {
                        'id': event_id,
                        'station': station,
                        'type': event_type,
                        'col_name': col
                    }
                    
                    events_in_order.append(event_data)
                    all_event_ids.add(event_id)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert event ID '{row[col]}' to integer in service {service['SrNum']}")
        
        # CRITICAL FIX: For UP services, we need to process events in TRAVERSAL order, not CSC column order
        if service['Dir'] == 'UP':
            # UP services traverse from From→To (e.g., VR→CCG), but CSC stores them in header order
            # We need to reorder events to match physical traversal
            service['Events'] = reorder_up_service_events(events_in_order, service['From'], service['To'])
            
            print(f"Service {service['SrNum']} (UP): Reordered events for {service['From']}→{service['To']}")
            if service['Events']:
                first_event = service['Events'][0]
                last_event = service['Events'][-1]
                print(f"  UP Traversal: {first_event['station']}({first_event['type'][0]}) → ... → {last_event['station']}({last_event['type'][0]})")
        else:
            # DOWN services are already in correct traversal order (CCG→VR)
            service['Events'] = events_in_order
            if service['Events']:
                first_event = service['Events'][0]
                last_event = service['Events'][-1]
                print(f"  DOWN Traversal: {first_event['station']}({first_event['type'][0]}) → ... → {last_event['station']}({last_event['type'][0]})")
        
        # Find first departure and last arrival
        for event in service['Events']:
            if event['type'] == 'departure' and service['FirstDeparture'] is None:
                service['FirstDeparture'] = event['id']
            if event['type'] == 'arrival':
                service['LastArrival'] = event['id']
        
        services.append(service)
        print(f"Service {service['SrNum']}: {service['From']}->{service['To']}, Dir: {service['Dir']}, Events: {len(service['Events'])}")
    
    return services, all_event_ids

def reorder_up_service_events(events_in_order, from_station, to_station):
    """Reorder UP service events to match physical traversal (From→To)"""
    # For UP services, we need to find the traversal path from From→To
    # The events in CSC are stored in header order, but traversal is reverse
    
    # Create mapping of station to events
    station_events = {}
    for event in events_in_order:
        if event['station'] not in station_events:
            station_events[event['station']] = []
        station_events[event['station']].append(event)
    
    # Determine the station sequence for this UP service
    try:
        from_idx = STATIONS_ORDER.index(from_station)
        to_idx = STATIONS_ORDER.index(to_station)
        
        if from_idx > to_idx:  # UP direction: from higher index to lower index
            traversal_stations = STATIONS_ORDER[to_idx:from_idx + 1][::-1]  # Reverse to get From→To
        else:
            traversal_stations = STATIONS_ORDER[from_idx:to_idx + 1]
    except ValueError:
        # Fallback: use the order they appear in events (shouldn't happen with valid data)
        traversal_stations = list(set([e['station'] for e in events_in_order]))
        traversal_stations.sort(key=lambda x: STATIONS_ORDER.index(x) if x in STATIONS_ORDER else 999)
        if from_station in traversal_stations and to_station in traversal_stations:
            from_idx = traversal_stations.index(from_station)
            to_idx = traversal_stations.index(to_station)
            traversal_stations = traversal_stations[from_idx:to_idx + 1]
    
    # Build events in traversal order
    traversal_events = []
    for station in traversal_stations:
        if station in station_events:
            # Add events for this station in correct order (arrival then departure)
            station_evts = station_events[station]
            arrivals = [e for e in station_evts if e['type'] == 'arrival']
            departures = [e for e in station_evts if e['type'] == 'departure']
            
            # For intermediate stations: arrival then departure
            # For origin: only departure
            # For destination: only arrival
            if station == from_station:
                traversal_events.extend(departures)  # Start with departure
            elif station == to_station:
                traversal_events.extend(arrivals)   # End with arrival
            else:
                # Intermediate: arrival then departure
                if arrivals:
                    traversal_events.append(arrivals[0])
                if departures:
                    traversal_events.append(departures[0])
    
    return traversal_events
def build_turnaround_link_pairs(constraints_data, services):
    """
    Terminal-only rake linkage pairs (dep_event, arr_event).

    Uses terminal events directly from JSON:
      - dep candidates come ONLY from turnaround[dep_dir]["dep"][speed]
      - arr candidates come ONLY from turnaround[arr_dir]["arr"][speed]

    Adds same-direction continuation + reverse-direction turnback.

    Always enforces station(dep) == station(arr).
    """

    # event_id -> station mapping from services
    event_station = {}
    for s in services:
        for ev in s["Events"]:
            event_station[ev["id"]] = ev["station"]

    link_pairs = []

    turnaround_data = constraints_data.get("turnaround", {})

    # Allow both reverse and same-direction linking
    pairing = [
        ("DOWN", "UP"),    # turnback
        ("UP", "DOWN"),    # turnback
        ("DOWN", "DOWN"),  # continuation
        ("UP", "UP")       # continuation
    ]

    for dep_dir, arr_dir in pairing:
        if dep_dir not in turnaround_data or arr_dir not in turnaround_data:
            continue

        dep_block = turnaround_data[dep_dir]["dep"]   # terminal departures
        arr_block = turnaround_data[arr_dir]["arr"]   # terminal arrivals

        for speed in dep_block.keys():  # fast, slow
            dep_ids = dep_block.get(speed, [])
            arr_ids = arr_block.get(speed, [])

            for d in dep_ids:
                st_d = event_station.get(d)
                if st_d is None:
                    continue

                for a in arr_ids:
                    st_a = event_station.get(a)
                    if st_a is None:
                        continue

                    if st_d == st_a:
                        link_pairs.append((d, a))

    return link_pairs





def create_traversal_constraints(model, services, travel_times):
    """Create traversal constraints with ZERO dwell time and perfect chronological order"""
    print("\nCreating traversal constraints (ZERO dwell)...")
    constraint_count = 0
    
    TRAVEL_TOLERANCE = 1e-5
    
    for service in services:
        events = service['Events']
        print(f"\nService {service['SrNum']} ({service['Dir']}): {service['From']}→{service['To']}")
        
        for i in range(len(events) - 1):
            current = events[i]
            next_ev = events[i + 1]
            
            print(f"  Event {i}: {current['station']}({current['type'][0]})[{current['id']}] → "
                  f"Event {i+1}: {next_ev['station']}({next_ev['type'][0]})[{next_ev['id']}]")
            
            # ZERO DWELL TIME: Departure immediately follows arrival at same station
            if (current['station'] == next_ev['station'] and 
                current['type'] == 'arrival' and 
                next_ev['type'] == 'departure'):
                
                # ZERO dwell time - depart immediately after arrival
                model.traversal_constraints.add(
                    model.t[next_ev['id']] - model.t[current['id']] == 0.0
                )
                constraint_count += 1
                print(f"    ZERO Dwell: {current['station']} = 0.0min")
            
            # Travel time constraint (departure -> arrival at different stations)
            elif (current['station'] != next_ev['station'] and 
                  current['type'] == 'departure' and 
                  next_ev['type'] == 'arrival'):
                
                segment = f"{current['station']}-{next_ev['station']}"
                travel_time = travel_times.get(service['PatNum'], {}).get(segment)
                
                if travel_time is not None:
                    # STRICT travel time equality
                    model.traversal_constraints.add(
                        model.t[next_ev['id']] - model.t[current['id']] == travel_time
                    )
                    constraint_count += 1
                    print(f"    Travel: {segment} = {travel_time}min")
            
            # PERFECT chronological order for ALL consecutive events
            # This ensures events happen in the correct sequence
            model.traversal_constraints.add(
                model.t[next_ev['id']] - model.t[current['id']] >= 0.0
            )
            constraint_count += 1
    
    print(f"Total traversal constraints: {constraint_count}")
    return constraint_count

def find_approximate_travel_time(from_station, to_station):
    """Find approximate travel time between stations based on station order"""
    try:
        from_idx = STATIONS_ORDER.index(from_station)
        to_idx = STATIONS_ORDER.index(to_station)
        distance = abs(to_idx - from_idx)
        
        # Approximate travel times based on distance
        if distance == 1:
            return 8.0  # Short hop
        elif distance == 2:
            return 15.0  # Medium hop
        elif distance == 3:
            return 22.0  # Long hop
        else:
            return distance * 7.0  # General formula
    except ValueError:
        return 10.0  # Default fallback

def create_headway_constraints(model, services, constraints_data):
    print("\nCreating headway constraints (USING JSON)...")

    headway_json = constraints_data["headway"]

    # Map event_id → station
    event_id_to_station = {}
    for service in services:
        for event in service["Events"]:
            event_id_to_station[event["id"]] = event["station"]

    # Group JSON-approved departure events by station and track type
    station_events = defaultdict(lambda: defaultdict(list))

    for direction in headway_json:
        for speed in headway_json[direction]:
            dep_ids = headway_json[direction][speed]
            track_type = f"{speed}_{direction}"

            for eid in dep_ids:
                if eid in event_id_to_station:
                    station = event_id_to_station[eid]
                    station_events[station][track_type].append(eid)

    # Create headway pairs
    headway_pairs = set()
    for station in station_events:
        for track_type in station_events[station]:
            dep_ids = station_events[station][track_type]
            for i in range(len(dep_ids)):
                for j in range(i + 1, len(dep_ids)):
                    headway_pairs.add((dep_ids[i], dep_ids[j]))

    print(f"Headway constraint pairs (JSON-based): {len(headway_pairs)}")

    # Create precedence variables
    model.HEADWAY_PAIRS = pyo.Set(initialize=list(headway_pairs))
    model.p = pyo.Var(model.HEADWAY_PAIRS, within=pyo.Binary)

    # Add constraints
    for (id1, id2) in model.HEADWAY_PAIRS:
        model.headway_constraints.add(
            model.t[id2] - model.t[id1] + (BIG_M + MIN_HEADWAY) * model.p[id1, id2] >= MIN_HEADWAY
        )
        model.headway_constraints.add(
            model.t[id2] - model.t[id1] + (BIG_M + MIN_HEADWAY) * model.p[id1, id2] <= BIG_M
        )

    return len(headway_pairs) * 2


def run_optimization():
    """Main optimization function"""
    # Load data
    csc_df, patterns_df, constraints_data = load_and_validate_data()
    
    # Calculate time horizon - ensure numeric types
    csc_df['Time'] = pd.to_numeric(csc_df['Time'], errors='coerce')
    min_time = float(csc_df['Time'].min())
    max_time = float(csc_df['Time'].max())
    time_horizon = max_time + 300
    print(f"Time range: {min_time} to {time_horizon} minutes")
    
    # Extract data
    travel_times = extract_travel_times(patterns_df)
    services, all_event_ids = build_service_sequences(csc_df)
    # --- Build metadata maps for events ---
    event_station = {}
    event_dir     = {}
    event_type    = {}
    event_srnum   = {}
    # --- Build terminal role maps ---
    is_first_dep = {}
    is_last_arr  = {}

    for s in services:
        fd = s.get("FirstDeparture", None)
        la = s.get("LastArrival", None)

        if fd is not None:
            is_first_dep[fd] = True
        if la is not None:
            is_last_arr[la] = True


    for s in services:
        for ev in s["Events"]:
            eid = ev["id"]
            event_station[eid] = ev["station"]
            event_dir[eid]     = s["Dir"]     # UP/DOWN
            event_type[eid]    = s["Type"]    # fast/slow
            event_srnum[eid]   = s["SrNum"]   # service number

    # Create model
    model = pyo.ConcreteModel()
    
     # Sets
    model.EVENTS = pyo.Set(initialize=list(all_event_ids))
    model.SERVICES = pyo.Set(initialize=range(len(services)))
    
    # Variables
    model.t = pyo.Var(model.EVENTS, bounds=(min_time - 10, time_horizon), within=pyo.NonNegativeReals)
    model.delta = pyo.Var(bounds=(2,10), within=pyo.NonNegativeReals)
   

    # --- Rake linking sets and variables ---
    link_pairs = build_turnaround_link_pairs(constraints_data, services)

    model.LINK_PAIRS = pyo.Set(initialize=link_pairs, dimen=2)
    model.X = pyo.Var(model.LINK_PAIRS, within=pyo.Binary)

    dep_events = sorted(set(i for (i, _) in link_pairs))
    arr_events = sorted(set(j for (_, j) in link_pairs))

    model.source = pyo.Var(dep_events, within=pyo.Binary)
    model.sink   = pyo.Var(arr_events, within=pyo.Binary)

    # Constraint lists
    model.start_constraints = pyo.ConstraintList()
    model.traversal_constraints = pyo.ConstraintList()
    model.headway_constraints = pyo.ConstraintList()
    model.distribution_constraints = pyo.ConstraintList()
    model.rake_constraints = pyo.ConstraintList()

   # 1. Start time constraints
    print("\nSetting start time constraints...")
    
    # Fix first service at 480
    first_service = next((s for s in services if s['SrNum'] == 1), None)
    if first_service and first_service['FirstDeparture']:
        model.start_constraints.add(model.t[first_service['FirstDeparture']] == 480.0)
        print(f"Fixed service 1 start at 480.0 minutes")
    
    # Other services close to original times
    for service in services:
        if service['FirstDeparture'] and service['SrNum'] != 1:
            model.start_constraints.add(
                model.t[service['FirstDeparture']] >= service['OriginalTime']          )
            model.start_constraints.add(
                model.t[service['FirstDeparture']] <= service['OriginalTime'] + 19
            )
    
    # 2. Traversal constraints
    traversal_count = create_traversal_constraints(model, services, travel_times)
    
    # 3. Headway constraints
    headway_count = create_headway_constraints(model, services, constraints_data)
    # 4.Turn around constraints
    

    # --- constraints container ---
    model.rake_constraints = pyo.ConstraintList()

    # each arrival either linked or sink
    for j in arr_events:
        model.rake_constraints.add(
            sum(model.X[i, j] for (i, jj) in model.LINK_PAIRS if jj == j)
            + model.sink[j] == 1
        )

    # each departure either linked or source
    for i in dep_events:
        model.rake_constraints.add(
            sum(model.X[i, j] for (ii, j) in model.LINK_PAIRS if ii == i)
            + model.source[i] == 1
        )

    

    # --- NEW: Turnaround time bounds ---
    TURN_LB = 5.0
    TURN_UB = 10.0

    for (i, j) in model.LINK_PAIRS:
        model.rake_constraints.add(
            model.t[i] - model.t[j] >= TURN_LB - BIG_M * (1 - model.X[i, j])
        )
        model.rake_constraints.add(
            model.t[i] - model.t[j] <= TURN_UB + BIG_M * (1 - model.X[i, j])
        )


    
    # 4. Distribution constraints (simplified)
    print("\nSetting distribution constraints...")
    dist_count = 0
    
    # Group by route
    route_services = defaultdict(list)
    for service in services:
        key = (service['From'], service['To'], service['Type'], service['Dir'])
        route_services[key].append(service)
    
    for route, route_services_list in route_services.items():
        if len(route_services_list) > 1:
            sorted_services = sorted(route_services_list, key=lambda x: x['OriginalTime'])
            ideal_spacing = 60.0 / len(sorted_services)
            
            for i in range(len(sorted_services) - 1):
                s1 = sorted_services[i]
                s2 = sorted_services[i + 1]
                
                if s1['FirstDeparture'] and s2['FirstDeparture']:
                    model.distribution_constraints.add(
                        model.t[s2['FirstDeparture']] - model.t[s1['FirstDeparture']] >= ideal_spacing - model.delta
                    )
                    model.distribution_constraints.add(
                        model.t[s2['FirstDeparture']] - model.t[s1['FirstDeparture']] <= ideal_spacing + model.delta
                    )
                    
                    dist_count += 1
    
    print(f"Distribution constraints: {dist_count}")
    
    RAKE_WEIGHT = 10000.0   # very large
    DELTA_WEIGHT = 1.0      # small

    def objective_rule(model):
        return (
            RAKE_WEIGHT * (
                sum(model.source[i] for i in dep_events)
                + sum(model.sink[j] for j in arr_events)
            )
            + DELTA_WEIGHT * model.delta
        )
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    
    # Solve
    print(f"\n{'='*60}")
    print("SOLVING OPTIMIZATION MODEL")
    print(f"{'='*60}")
    print(f"Events: {len(all_event_ids)}")
    print(f"Services: {len(services)}")
    print(f"Traversal constraints: {traversal_count}")
    print(f"Headway constraints: {headway_count}")
    print(f"Distribution constraints: {dist_count}")
    print(f"Time limit: {NEW_TIME_LIMIT}s")
    
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = NEW_TIME_LIMIT
    solver.options['MIPGap'] = NEW_MIP_GAP
    solver.options['FeasibilityTol'] = FEASIBILITY_TOL
    solver.options['OptimalityTol'] = 1e-5 
    try:
        results = solver.solve(model, tee=True)
    except Exception as e:
        print(f"Solver error: {e}")
        return None, None, None,None, None, None, None, None, None, None, None
    
    # Process results
    if results.solver.termination_condition in [pyo.TerminationCondition.optimal, 
                                              pyo.TerminationCondition.feasible]:
        print("\n✓ SOLUTION FOUND!")
        # --- Extract rake linkages ---
        active_links = [
            (i, j) for (i, j) in model.LINK_PAIRS
            if model.X[i, j].value is not None and model.X[i, j].value > 0.5
        ]

        print(f"Active rake links: {len(active_links)}")
        # --- Rake consistency sanity check ---
        rake_count = sum(model.source[i].value for i in dep_events)
        print(f"Minimum rakes used in peak window: {rake_count}")
        
        
        # Extract times
        event_times = {}
        for event_id in all_event_ids:
            if model.t[event_id].value is not None:
                event_times[event_id] = round(float(model.t[event_id].value), 2)
        print("\nActive rake links (ARR → DEP):")

        for (dep, arr) in active_links:
            # dep = departure event id (next service)
            # arr = arrival event id (previous service)

            stn = event_station.get(dep, "NA")   # terminal station

            dep_dir  = event_dir.get(dep, "NA")
            arr_dir  = event_dir.get(arr, "NA")
            dep_type = event_type.get(dep, "NA")  # fast/slow

            dep_svc = event_srnum.get(dep, "NA")
            arr_svc = event_srnum.get(arr, "NA")

            t_dep = event_times.get(dep, None)
            t_arr = event_times.get(arr, None)

            # turnaround time
            if t_dep is not None and t_arr is not None:
                delta = t_dep - t_arr
                delta_str = f"{delta:.1f}"
            else:
                delta_str = "NA"

            print(
                f"[{stn}] "
                f"{arr} (Arr, S{arr_svc}, {arr_dir}) @ {t_arr}  →  "
                f"{dep} (Dep, S{dep_svc}, {dep_dir}, {dep_type}) @ {t_dep}  "
                f"Δ={delta_str}"
            )


        min_time = min(event_times.values()) if event_times else None
        max_time = max(event_times.values()) if event_times else None

        return model, event_times, csc_df, services, min_time, max_time, active_links, event_station, is_first_dep, is_last_arr,event_type

         
        

    else:
        print(f"\n✗ Optimization failed: {results.solver.termination_condition}")
        return None, None, None, None, None, None,None,None,None,None,None
def extract_constraints(model, filename="constraints_dump.txt"):
    with open(filename, "w") as f:
        for cname in model.component_map(pyo.Constraint, active=True):
            constr_block = getattr(model, cname)
            f.write(f"\n### Constraint Block: {cname} ###\n")

            for idx in constr_block:
                c = constr_block[idx]

                expr = str(c.body)

                if c.lower is not None and c.upper is not None:
                    if abs(c.upper - c.lower) < 1e-6:
                        f.write(f"{expr} = {c.upper}\n")
                    else:
                        f.write(f"{c.lower} <= {expr} <= {c.upper}\n")
                elif c.lower is not None:
                    f.write(f"{expr} >= {c.lower}\n")
                elif c.upper is not None:
                    f.write(f"{expr} <= {c.upper}\n")

def generate_optimized_schedule(event_times, original_csc_df):
    """Generate optimized schedule CSV"""
    optimized_df = original_csc_df.copy()
    optimized_df.rename(columns={'Time': 'Original_Time'}, inplace=True)
    optimized_df['Optimized_Start_Time'] = np.nan
    
    # Ensure numeric types
    for col in OUTPUT_COLS_STATION_EVENTS:
        optimized_df[col] = pd.to_numeric(optimized_df[col], errors='coerce')
    
    # Replace event IDs with times
    for col in OUTPUT_COLS_STATION_EVENTS:
        def map_event_to_time(val):
            if pd.notna(val) and val != '':
                try:
                    event_id = int(float(val))  # Handle both int and float strings
                    return event_times.get(event_id, np.nan)
                except (ValueError, TypeError):
                    return np.nan
            return np.nan
        
        optimized_df[col] = optimized_df[col].apply(map_event_to_time)
    
    # Calculate start times
    for idx, row in optimized_df.iterrows():
        times = []
        for col in OUTPUT_COLS_STATION_EVENTS:
            val = row[col]
            if pd.notna(val) and not np.isnan(val):
                try:
                    times.append(float(val))
                except (ValueError, TypeError):
                    continue
        if times:
            optimized_df.at[idx, 'Optimized_Start_Time'] = min(times)
    
    # Sort by optimized start time
    optimized_df = optimized_df.sort_values('Optimized_Start_Time').reset_index(drop=True)
    
    return optimized_df

def plot_time_vs_distance(optimized_df):
    """
    Create time vs distance graph with stations on X-axis and time on Y-axis
    This shows the progression of trains along the route over time
    """
    print("\nGenerating time vs distance graph...")
    
    track_types = [
        ('fast', 'UP', 'red', 'Fast UP'),
        ('fast', 'DOWN', 'blue', 'Fast DOWN'), 
        ('slow', 'UP', 'green', 'Slow UP'),
        ('slow', 'DOWN', 'orange', 'Slow DOWN')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    # Create station positions for X-axis
    station_positions = {station: i for i, station in enumerate(STATIONS_ORDER)}
    
    for idx, (svc_type, direction, color, title) in enumerate(track_types):
        ax = axes[idx]
        
        services = optimized_df[
            (optimized_df['Type'] == svc_type) & 
            (optimized_df['Dir'] == direction)
        ]
        
        print(f"Plotting {title}: {len(services)} services")
        
        # Plot each service
        for _, service in services.iterrows():
            times, stations_visited = [], []
            
            # Collect all events for this service
            for station in STATIONS_ORDER:
                dep_col = station + 'd'
                arr_col = station + 'a'
                
                dep_time = service.get(dep_col)
                arr_time = service.get(arr_col)
                
                # Handle NaN values and ensure numeric types
                if pd.notna(dep_time) and dep_time != '':
                    try:
                        times.append(float(dep_time))
                        stations_visited.append(station)
                    except (ValueError, TypeError):
                        pass
                elif pd.notna(arr_time) and arr_time != '':
                    try:
                        times.append(float(arr_time))
                        stations_visited.append(station)
                    except (ValueError, TypeError):
                        pass
            
            if len(times) > 1:
                # Convert stations to numerical positions
                x_positions = [station_positions[stn] for stn in stations_visited]
                
                # Plot the service trajectory
                ax.plot(x_positions, times, marker='o', linewidth=2, color=color, alpha=0.7, markersize=4)
                
                # Add service number annotation at start point
                if len(x_positions) > 0:
                    try:
                        service_num = int(service['SrNum'])  # Ensure integer
                        ax.annotate(f"S{service_num}", 
                                   (x_positions[0], times[0]),
                                   textcoords="offset points", 
                                   xytext=(5,5), 
                                   ha='left', 
                                   fontsize=8, 
                                   alpha=0.8)
                    except (ValueError, TypeError):
                        pass  # Skip annotation if service number is not numeric
        
        # Configure plot
        # Axis labels
        ax.set_xlabel('Station', fontsize=20, fontweight='bold')
        ax.set_ylabel('Time (minutes)', fontsize=20, fontweight='bold')

        # Title
        ax.set_title(f'{title} Services\n({len(services)} services)', fontsize=22, fontweight='bold')

        # X-axis tick labels (station names)
        ax.set_xticks(range(len(STATIONS_ORDER)))
        ax.set_xticklabels(STATIONS_ORDER, rotation=45, ha='right', fontsize=16, fontweight='bold')

        # Y-axis tick labels (time values)
        ax.tick_params(axis='y', which='major', labelsize=16)
        ax.tick_params(axis='x', which='major', labelsize=16)

        
        # Set y-axis limits based on data
        all_times = []
        for col in OUTPUT_COLS_STATION_EVENTS:
            if col in optimized_df.columns:
                col_times = optimized_df[col].dropna()
                if len(col_times) > 0:
                    # Convert to float and filter out non-numeric values
                    numeric_times = []
                    for time_val in col_times:
                        try:
                            numeric_times.append(float(time_val))
                        except (ValueError, TypeError):
                            continue
                    if numeric_times:
                        all_times.extend(numeric_times)
        
        if all_times:
            ax.set_ylim(min(all_times) - 10, max(all_times) + 10)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_vs_distance_graph.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved time_vs_distance_graph.png")

def plot_track_type_time_vs_station(optimized_df, min_time, max_time):
    """
    Plot time vs station for each track type, inverting the Y-axis 
    for UP services to ensure visualization of traversal from terminus (top) 
    to city center (bottom).
    """
    track_types = [
        ('fast', 'UP', 'red', 'Fast UP'),
        ('fast', 'DOWN', 'blue', 'Fast DOWN'), 
        ('slow', 'UP', 'green', 'Slow UP'),
        ('slow', 'DOWN', 'orange', 'Slow DOWN')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    # Use the full STATIONS_ORDER for a consistent index map
    all_stations_for_indexing = STATIONS_ORDER
    station_idx = {s: i for i, s in enumerate(all_stations_for_indexing)}
    
    # Find the range of numerical indices (0 to len-1)
    min_plot_idx = min(station_idx.values())
    max_plot_idx = max(station_idx.values())
    
    for idx, (svc_type, direction, color, title) in enumerate(track_types):
        ax = axes[idx]
        
        services = optimized_df[
            (optimized_df['Type'] == svc_type) & 
            (optimized_df['Dir'] == direction)
        ]
        
        # Plot each service
        for _, service in services.iterrows():
            times, stns = [], []
            
            # Use the full station list to check for event times
            for station in all_stations_for_indexing:
                if station not in station_idx: continue
                    
                dep_col = station + 'd'
                arr_col = station + 'a'
                
                dep_time = service.get(dep_col)
                arr_time = service.get(arr_col)
                
                # Check for dep or arr time (ensure numeric)
                if pd.notna(dep_time) and dep_time != '':
                    try:
                        times.append(float(dep_time))
                        stns.append(station_idx[station])
                    except (ValueError, TypeError):
                        pass
                elif pd.notna(arr_time) and arr_time != '':
                    try:
                        times.append(float(arr_time))
                        stns.append(station_idx[station])
                    except (ValueError, TypeError):
                        pass
            
            if len(times) > 1:
                # Plotting uses the numerical index (stns)
                ax.plot(times, stns, marker='.', linewidth=1.5, color=color, alpha=0.7)
        
        # --- Configure Plot and AXES ---
        
        # Set Y-axis labels using the full station list
        ax.set_yticks(range(len(all_stations_for_indexing)))
        ax.set_yticklabels(all_stations_for_indexing)

        # CRITICAL CHANGE: INVERT Y-AXIS BASED ON DIRECTION
        if direction == 'UP':
            # UP services (VR/DRD -> CCG) should flow visually from top to bottom.
            # Inverts the axis so that the high index (VR/DRD) is at the top (y=0).
            ax.set_ylim(max_plot_idx + 0.5, min_plot_idx - 0.5)

        else: # DOWN services (CCG -> VR/DRD)
            # DOWN services (CCG -> DRD) naturally flow top-to-bottom with standard indexing.
            ax.set_ylim(min_plot_idx - 0.5, max_plot_idx + 0.5)
            
        ax.set_title(f'{title} Services\n({len(services)} services)')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Station')
        ax.grid(True, alpha=0.3)
        
        # Set time range
        all_times_flat = []
        for col in optimized_df.filter(regex='a|d$').columns:
            col_times = optimized_df[col].dropna()
            for time_val in col_times:
                try:
                    all_times_flat.append(float(time_val))
                except (ValueError, TypeError):
                    continue
        
        if all_times_flat:
            ax.set_xlim(min(all_times_flat) - 10, max(all_times_flat) + 10)
    
    plt.tight_layout()
    plt.savefig('optimized_time_vs_station_by_track_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved optimized_time_vs_station_by_track_type.png with UP axis corrected.")

def plot_time_vs_distance_with_linkages(optimized_df, active_links, event_times, event_station,
                                        is_first_dep, is_last_arr, event_type):
    """
    Two plots: FAST and SLOW
    - UP+DOWN together
    - Only plot linkages belonging to that track type
    - Only plot solver-chosen linkages which are terminal-to-terminal:
         LastArrival(arr) -> FirstDeparture(dep)
    """

    print("\nGenerating time vs distance graph with SOLVER terminal linkages (correct fast/slow split)...")

    track_panels = [
        ("fast", "FAST Track: UP + DOWN with Solver Linkages"),
        ("slow", "SLOW Track: UP + DOWN with Solver Linkages")
    ]

    dir_colors = {"UP": "red", "DOWN": "blue"}
    TURN_UB_PLOT = 15.0

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    axes = axes.flatten()

    station_positions = {station: i for i, station in enumerate(STATIONS_ORDER)}

    for idx, (panel_type, title) in enumerate(track_panels):
        ax = axes[idx]

        # plot services of this type
        services_df = optimized_df[optimized_df["Type"] == panel_type]

        for _, service in services_df.iterrows():
            direction = service.get("Dir", None)
            if direction not in ["UP", "DOWN"]:
                continue

            color = dir_colors[direction]

            times, stations_visited = [], []
            for station in STATIONS_ORDER:
                dep_col = station + "d"
                arr_col = station + "a"

                dep_time = service.get(dep_col)
                arr_time = service.get(arr_col)

                if pd.notna(dep_time):
                    times.append(float(dep_time))
                    stations_visited.append(station)
                elif pd.notna(arr_time):
                    times.append(float(arr_time))
                    stations_visited.append(station)

            if len(times) > 1:
                x_positions = [station_positions[s] for s in stations_visited]
                ax.plot(x_positions, times, linewidth=1.2, color=color, alpha=0.5)

        # plot linkages only for this panel_type
        station_link_count = defaultdict(int)
        arrows_plotted = 0

        for dep, arr in active_links:

            # ✅ filter by track type correctly
            if event_type.get(dep, None) != panel_type:
                continue

            # ✅ terminal-to-terminal only
            if not is_last_arr.get(arr, False):
                continue
            if not is_first_dep.get(dep, False):
                continue

            stn = event_station.get(dep, None)
            if stn is None or stn not in station_positions:
                continue

            if event_station.get(arr, None) != stn:
                continue

            t_dep = event_times.get(dep, None)
            t_arr = event_times.get(arr, None)

            if t_dep is None or t_arr is None:
                continue

            turnaround = t_dep - t_arr
            if turnaround < 0 or turnaround > TURN_UB_PLOT:
                continue

            x_base = station_positions[stn]
            station_link_count[stn] += 1
            offset = (station_link_count[stn] % 6) * 0.06
            x = x_base + offset

            ax.annotate(
                "",
                xy=(x, t_dep),
                xytext=(x, t_arr),
                arrowprops=dict(
                    arrowstyle="->",
                    linestyle="--",
                    lw=2,
                    color="black",
                    alpha=0.85
                )
            )
            ax.text(x + 0.03, (t_arr + t_dep) / 2, f"Δ={turnaround:.0f}",
                    fontsize=9, alpha=0.9)

            arrows_plotted += 1

        print(f"✅ {panel_type.upper()} arrows drawn = {arrows_plotted}")

        # legend
        ax.plot([], [], color="red", label="UP services")
        ax.plot([], [], color="blue", label="DOWN services")
        ax.plot([], [], linestyle="--", color="black",
                label=f"Solver linkage ({panel_type})")

        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_xlabel("Station")
        ax.set_ylabel("Time (minutes)")
        ax.set_xticks(range(len(STATIONS_ORDER)))
        ax.set_xticklabels(STATIONS_ORDER, rotation=45, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("time_vs_distance_solver_linkages_fast_slow_split.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved time_vs_distance_solver_linkages_fast_slow_split.png")




# Main execution
if __name__ == '__main__':
    try:
        print("Starting optimization...")
        
        model, event_times, csc_df, services, min_time, max_time, active_links, event_station, is_first_dep, is_last_arr,event_type = run_optimization()

        if model is None:
            raise ValueError("Optimization failed")

        
        extract_constraints(model, filename="constraints_dump.txt")
        print("constraints_dump.txt generated")

        print("\nGenerating optimized schedule...")
        optimized_df = generate_optimized_schedule(event_times, csc_df)
        optimized_df.to_csv('optimized_schedule_f_s.csv', index=False)
        print(" Saved optimized_schedule_f_s.csv")

        print("\nGenerating plots...")
        plot_time_vs_distance(optimized_df)
        plot_track_type_time_vs_station(optimized_df, min_time, max_time)
        plot_time_vs_distance_with_linkages(optimized_df, active_links, event_times, event_station, is_first_dep, is_last_arr,  event_type)

        

        print("\n All tasks completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
