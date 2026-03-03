from flask import Flask, request, jsonify
import sys
sys.path.append('docs/lab_simulators')

from decision_quality_simulator import DecisionQualitySimulator, simple_stage_factory
from otv_physics_simulator import OTVPhysicsSimulator, Material

app = Flask(__name__)

# Set up simple default stages for the decision-quality simulator
stage1 = simple_stage_factory({
    'dielectric_constant': (3.0, True),
    'loss_tangent': (0.015, False),
})
stage2 = simple_stage_factory({
    'moisture_uptake': (0.02, False),
})

@app.route('/decision', methods=['GET', 'POST'])
def decision_sim():
    # Number of candidates to simulate (defaults to 10)
    n = int(request.args.get('n_candidates', (request.get_json() or {}).get('n_candidates', 10)))
    sim = DecisionQualitySimulator(stages=[stage1, stage2])
    df = sim.run(n_candidates=n)
    return df.to_json(orient='records'), 200, {'Content-Type': 'application/json'}

@app.route('/otv', methods=['POST'])
def otv_sim():
    # Expect JSON with a list of material definitions
    data = request.get_json(force=True)
    materials = []
    for m in data.get('materials', []):
        materials.append(Material(
            dielectric_constant=float(m['dielectric_constant']),
            loss_tangent=float(m['loss_tangent']),
            moisture_uptake=float(m['moisture_uptake']),
            adhesion_energy=float(m['adhesion_energy']),
            via_yield=float(m['via_yield']),
            ion_content=float(m['ion_content']),
            name=m.get('name', 'material')
        ))
    sim = OTVPhysicsSimulator()
    df = sim.run(materials)
    return df.to_json(orient='records'), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
