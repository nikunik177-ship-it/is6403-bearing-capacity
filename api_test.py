import urllib.request, json

# Health check
with urllib.request.urlopen('http://127.0.0.1:8000/api/health') as r:
    print('Health:', json.loads(r.read()))

# API test
payload = json.dumps({
    'cohesion': 0, 'friction_angle': 30, 'unit_weight': 18,
    'sat_unit_weight': 20, 'water_table_depth': 999,
    'footing_shape': 'strip', 'width': 1.5, 'depth': 1.5,
    'load_inclination': 0, 'failure_mode': 'general', 'fos': 2.5
}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:8000/api/calculate',
    data=payload,
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())

print("--- Arora Strip Footing (phi=30, c=0, Df=1.5m, B=1.5m) ---")
print("q_nu  =", round(data['q_nu'], 2), "kPa")
print("q_ns  =", round(data['q_ns'], 2), "kPa  (FoS={})".format(data['fos']))
print("q_s   =", round(data['q_s'],  2), "kPa")
print("Mode  =", data['failure_mode_used'])
print("Nc={}, Nq={}, Ng={}".format(data['N_c'], data['N_q'], data['N_gamma']))
print("API: OK")
