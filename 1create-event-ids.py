import csv
import json
import yaml

def debug_print(message, data=None):
    print(f"\nDEBUG: {message}")
    if data is not None:
        print(data)

# --- Initialize constraints structure ---
constraints = {
    "headway": {
        "UP": {"fast": [], "slow": []},
        "DOWN": {"fast": [], "slow": []}
    },
    "turnaround": {
        "UP": {"dep": {"fast": [], "slow": []}, "arr": {"fast": [], "slow": []}},
        "DOWN": {"dep": {"fast": [], "slow": []}, "arr": {"fast": [], "slow": []}}
    },
    "link": {
        "UP": {"fast": [], "slow": []},
        "DOWN": {"fast": [], "slow": []}
    }
}

# --- Load stations ---
with open("WR_supply.csv", "r") as f:
    reader = list(csv.reader(f))
    stations = [col.strip() for col in reader[1][2:13] if col.strip()]

valid_stations = set(stations)

# --- time_table_transpose_df header ---
header = ["SrNum", "Time", "Type", "Dir", "PatNum", "From", "To"]
for st in stations:
    header.append(f"{st}a")
    header.append(f"{st}d")

# ✅ Optimization: O(1) lookup
col_index = {col: idx for idx, col in enumerate(header)}

time_table_transpose_df = [header]

# --- Load patterns ---
pattern_times = {}
with open("OWR-patterns-consolidated-sequential.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pat_id = row["Pattern_ID"].strip()
        raw_segments = []

        for i in range(1, 11):
            seg = row.get(f"Major_Segment_{i}")
            time = row.get(f"Time_{i}")
            if seg and time and seg.strip() and time.strip():
                try:
                    s_from, s_to = seg.strip().split("-")
                    t = int(float(time))
                    raw_segments.append((s_from.strip(), s_to.strip(), t))
                except:
                    continue

        # Aggregate between valid stations only
        segments = []
        current_valid = None
        current_time = 0

        for u, v, t in raw_segments:
            current_time += t

            if u in valid_stations and current_valid is None:
                current_valid = u
                current_time = t

            if v in valid_stations and current_valid is not None:
                segments.append((current_valid, v, current_time))
                current_valid = v
                current_time = 0

        if segments:
            pattern_times[pat_id] = segments

debug_print("Patterns loaded", len(pattern_times))

# --- Helper: Compute travel path ---
def get_travel_path(pat_segments, frm, to):
    path = []
    time = 0
    collecting = False

    for u, v, t in pat_segments:
        if u == frm:
            collecting = True
            path.append((u, time, 'd'))  # departure at frm

        if collecting:
            time += t
            # arrival at v
            path.append((v, time, 'a'))

            if v == to:
                break

            # departure at v (intermediate)
            path.append((v, time, 'd'))

    if not path or path[-1][0] != to:
        return [], 0

    return path, time

# --- Load services ---
wishlist = []
with open("Oservices_data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        wishlist.append(
            (int(row["SrNum"]), int(row["Time"]),
             row["Type"].strip(), row["Dir"].strip(),
             row["PatNum"].strip(), row["From"].strip(), row["To"].strip())
        )

debug_print("Services loaded", len(wishlist))

# --- Build time_table_transpose_df and constraints ---
dep_id = 20000
arr_id = 100

built = 0
skipped_invalid_station = 0
skipped_pattern_missing = 0
skipped_path_invalid = 0

for sr_num, dep_time, typ, direc, pat, frm, to in wishlist:

    if frm not in valid_stations or to not in valid_stations:
        skipped_invalid_station += 1
        continue

    pat_segments = pattern_times.get(pat)
    if not pat_segments:
        skipped_pattern_missing += 1
        continue

    path, travel_time = get_travel_path(pat_segments, frm, to)
    if not path:
        skipped_path_invalid += 1
        continue

    row = [""] * len(header)
    row[col_index["SrNum"]] = sr_num
    row[col_index["Time"]] = dep_time
    row[col_index["Type"]] = typ
    row[col_index["Dir"]] = direc
    row[col_index["PatNum"]] = pat
    row[col_index["From"]] = frm
    row[col_index["To"]] = to

    current_dep_id = dep_id
    current_arr_id = arr_id

    dep_ids_for_this_train = []
    terminal_dep_id = None
    terminal_arr_id = None

    for station, t, action in path:
        col = f"{station}{action}"
        if col not in col_index:
            continue

        if action == "d":
            row[col_index[col]] = str(current_dep_id)
            dep_ids_for_this_train.append(current_dep_id)

            # ✅ terminal dep = first departure at From
            if station == frm and terminal_dep_id is None:
                terminal_dep_id = current_dep_id

            current_dep_id += 1

        else:
            row[col_index[col]] = str(current_arr_id)

            # ✅ terminal arr = arrival at To
            if station == to:
                terminal_arr_id = current_arr_id

            current_arr_id += 1

    time_table_transpose_df.append(row)
    built += 1

    # --- Constraints ---
    # headway: keep all departure events
    constraints["headway"][direc][typ].extend(dep_ids_for_this_train)

    # turnaround: store only terminal dep + terminal arr
    if terminal_dep_id is not None:
        constraints["turnaround"][direc]["dep"][typ].append(terminal_dep_id)

    if terminal_arr_id is not None:
        constraints["turnaround"][direc]["arr"][typ].append(terminal_arr_id)

    dep_id = current_dep_id
    arr_id = current_arr_id

# --- Save time_table_transpose_df ---
with open("1-o-event-ids.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(time_table_transpose_df)

# --- Save constraints ---
with open("constraints_f_s.json", "w") as f:
    json.dump(constraints, f, indent=4)

with open("constraints_f_s.yaml", "w") as f:
    yaml.dump(constraints, f, default_flow_style=False)

# --- Debug ---
debug_print("Built services", built)
debug_print("Skipped invalid stations", skipped_invalid_station)
debug_print("Skipped missing pattern", skipped_pattern_missing)
debug_print("Skipped invalid path", skipped_path_invalid)

debug_print("Turnaround dep counts", {
    d: {t: len(constraints["turnaround"][d]["dep"][t]) for t in constraints["turnaround"][d]["dep"]}
    for d in constraints["turnaround"]
})
debug_print("Turnaround arr counts", {
    d: {t: len(constraints["turnaround"][d]["arr"][t]) for t in constraints["turnaround"][d]["arr"]}
    for d in constraints["turnaround"]
})

print("\nDONE ✅")
