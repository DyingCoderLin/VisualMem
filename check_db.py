import sqlite3, os
db_path = os.path.join('visualmem_storage', 'visualmem_activity.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Check assignments status
print("=== Activity Assignments ===")
c.execute("SELECT cluster_status, COUNT(*) FROM activity_assignments GROUP BY cluster_status")
for r in c.fetchall():
    print(f'  {r[0]}: {r[1]}')

# Check if assignments have cluster_id
c.execute("SELECT COUNT(*) FROM activity_assignments WHERE activity_cluster_id IS NOT NULL")
print(f'\nAssignments with cluster_id: {c.fetchone()[0]}')
c.execute("SELECT COUNT(*) FROM activity_assignments WHERE activity_cluster_id IS NULL")
print(f'Assignments without cluster_id: {c.fetchone()[0]}')

# Check pending groups
print("\n=== Pending Groups ===")
c.execute("SELECT resolved_label, resolved_cluster_id IS NOT NULL as has_id, COUNT(*) FROM pending_groups GROUP BY resolved_label, resolved_cluster_id IS NOT NULL")
for r in c.fetchall():
    print(f'  label={r[0]}, has_cluster_id={r[1]}, count={r[2]}')

# Check sample assignments
print("\n=== Sample Assignments ===")
c.execute("SELECT app_name, activity_label, provisional_label, cluster_status, activity_cluster_id FROM activity_assignments LIMIT 20")
for r in c.fetchall():
    print(f'  app={r[0]}, label={r[1]}, prov={r[2]}, status={r[3]}, cid={r[4]}')

# Check frame timestamps
c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM activity_assignments")
r = c.fetchone()
print(f"\nAssignment time range: {r[0]} to {r[1]}")

conn.close()
