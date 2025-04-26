from collections import defaultdict

# Read the airfoil names
with open('unique_airfoil_names.txt', 'r') as f:
    airfoil_names = [line.strip() for line in f.readlines()]

# Function to extract prefix
def get_prefix(name):
    # Remove any trailing numbers or letters that are part of the version
    base = ''.join([c for c in name if not c.isdigit()])
    # Remove any trailing 'sm', 'mod', etc.
    for suffix in ['sm', 'mod']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return base

# Group airfoils by prefix
prefix_groups = defaultdict(list)
for name in airfoil_names:
    prefix = get_prefix(name)
    prefix_groups[prefix].append(name)

# Sort prefixes by number of airfoils
sorted_prefixes = sorted(prefix_groups.items(), key=lambda x: len(x[1]), reverse=True)

# Print results
print("Airfoil Groups by Prefix (sorted by count):")
print("=" * 50)
for prefix, airfoils in sorted_prefixes:
    if len(airfoils) > 1:  # Only show prefixes with multiple airfoils
        print(f"\nPrefix: {prefix}")
        print(f"Count: {len(airfoils)}")
        print("Examples:", ', '.join(airfoils[:5]) + ('...' if len(airfoils) > 5 else ''))

# Print single-instance prefixes
single_prefixes = [prefix for prefix, airfoils in prefix_groups.items() if len(airfoils) == 1]
print("\nSingle-instance prefixes:", len(single_prefixes))
print("Examples:", ', '.join(single_prefixes[:10]) + ('...' if len(single_prefixes) > 10 else '')) 