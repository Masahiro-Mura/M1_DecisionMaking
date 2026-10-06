"""Read-only production diagnostics: A learned, B zero, C +0.1, D -0.1 rad.

Run: python diagnostics/testMPCReference.py --output diagnostics/mpcReference
Each case runs one complete deterministic episode using the real env.step.
Instance-local wrappers capture results without replacing control calculations.
Manual cases use the inverse of the existing adapter, verified each step.
No policy, reward, parameters, or production files are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from highway_env.vehicle.kinematics import Vehicle
from cbf.roadCBF import RoadCBFContext, left_road_barrier, right_road_barrier
from cbf.safeController import SafeController
from environment.actionAdapter import action_to_reference
from environment.mergeEnvironment import MergeEnvironment
from environment.observation import lane_change_feasibility
from mpc.kinematicBicycleModel import discrete_step
from mpc.mpcController import MPCController
from parameters.environmentParameters import (
    REFERENCE_SPEED_MIN, REFERENCE_SPEED_MAX,
    REFERENCE_HEADING_MIN, REFERENCE_HEADING_MAX, SIMULATION_DT,
)
from parameters.mpcParameters import Q, R, CONTROL_LOWER_BOUND, CONTROL_UPPER_BOUND
from rl.ppoAgent import PPOAgent


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def normalized_reference(speed, heading):
    return np.array([
        2 * (speed - REFERENCE_SPEED_MIN) / (REFERENCE_SPEED_MAX - REFERENCE_SPEED_MIN) - 1,
        2 * (heading - REFERENCE_HEADING_MIN) / (REFERENCE_HEADING_MAX - REFERENCE_HEADING_MIN) - 1,
    ])


def cost_components(result, reference):
    errors = reference - result.predicted_states[:-1, 2:4]
    controls = result.control_sequence
    parts = {
        'cost_speed': float(np.sum(Q[0, 0] * errors[:, 0] ** 2)),
        'cost_heading': float(np.sum(Q[1, 1] * errors[:, 1] ** 2)),
        'cost_tracking_cross': float(np.sum((Q[0, 1] + Q[1, 0]) * errors[:, 0] * errors[:, 1])),
        'cost_acceleration': float(np.sum(R[0, 0] * controls[:, 0] ** 2)),
        'cost_steering': float(np.sum(R[1, 1] * controls[:, 1] ** 2)),
        'cost_control_cross': float(np.sum((R[0, 1] + R[1, 0]) * controls[:, 0] * controls[:, 1])),
    }
    np.testing.assert_allclose(sum(parts.values()), result.objective_value, rtol=1e-10, atol=1e-9)
    return parts


def isolated_checks():
    """Checks isolate MPC/dynamics/CBF from learned behavior and traffic."""
    rows = []
    for speed in (5., 10.):
        state = np.array([230., 8., speed, 0.])
        for heading in (0., .1, -.1):
            controller = MPCController()
            result = controller.solve(state, speed, heading)
            assert result.success, result.message
            if heading == 0:
                assert abs(result.control[1]) < 1e-5
            else:
                assert result.control[1] * heading > 0
            vehicle = Vehicle(None, state[:2].copy(), heading=0., speed=speed)
            vehicle.act({'acceleration': float(result.control[0]), 'steering': float(result.control[1])})
            vehicle.step(SIMULATION_DT)
            actual = np.array([*vehicle.position, vehicle.speed, vehicle.heading])
            np.testing.assert_allclose(actual, discrete_step(state, result.control), atol=1e-12)
            rows.append({'speed': speed, 'theta_ref': heading, 'steering': float(result.control[1]), 'passed': True})
        # A previously saturated sequence must not force a zero-reference solution.
        controller = MPCController()
        controller._previous_solution = np.tile(CONTROL_LOWER_BOUND, (controller.prediction_horizon, 1))
        result = controller.solve(state, speed, 0.)
        assert result.success and abs(result.control[1]) < 1e-3
        rows.append({'speed': speed, 'theta_ref': 0., 'saturated_warm_start_steering': float(result.control[1]), 'passed': True})
        for ego_heading in (-.2, .2):
            state[3] = ego_heading
            for offset in (0., .1, -.1):
                result = MPCController().solve(state, speed, ego_heading + offset)
                assert result.success, result.message
                if offset == 0:
                    assert abs(result.control[1]) < 1e-5
                else:
                    assert result.control[1] * offset > 0
                rows.append({'speed': speed, 'ego_heading': ego_heading,
                             'theta_ref': ego_heading + offset,
                             'steering': float(result.control[1]), 'passed': True})
    context = RoadCBFContext(230., 8., 0., 0.)
    controller = SafeController()
    safe = controller.correct(np.array([230., 8., 5., 0.]), np.zeros(2), [], context, True, True)
    assert safe.success and np.linalg.norm(safe.correction) < 1e-6
    # IMPLEMENTATION ASSUMPTION: synthetic test near h1 boundary with no traffic;
    # this fixture tests intervention, not the paper's choice of active boundary.
    edge = RoadCBFContext(230., 8., 1.99, 0.)
    unsafe = controller.correct(np.array([230., 8., 5., 0.]), np.array([0., .3]), [], edge, True, True)
    assert unsafe.success and np.linalg.norm(unsafe.correction) > 1e-3
    assert unsafe.minimum_constraint_value >= -1e-6
    rows.append({'safe_cbf_correction': safe.correction.tolist(), 'unsafe_cbf_correction': unsafe.correction.tolist(), 'passed': True})
    for action in (-1., 0., 1.):
        ref = action_to_reference(np.array([0., action]))
        np.testing.assert_allclose(ref.heading, action * np.pi / 4, atol=1e-14)
    return rows


def run_case(agent, case, fixed_heading, args, output):
    env = MergeEnvironment(ego_initial_speed=args.speed)
    observation, _ = env.reset(seed=args.seed)
    capture = {}
    solve = env.mpc.solve
    correct = env.safe_controller.correct

    def capture_solve(*pos, **kwargs):
        capture['initial_guess'] = env.mpc._initial_guess().copy()
        result = solve(*pos, **kwargs)
        capture['mpc'] = result
        return result

    def capture_correct(*pos, **kwargs):
        constraints = env.safe_controller._build_constraints(**kwargs)
        capture['nominal_constraint_min'] = min(
            (float(c['fun'](np.zeros(2))) for c in constraints), default=float('inf'))
        result = correct(*pos, **kwargs)
        capture['cbf'] = result
        return result

    env.mpc.solve = capture_solve
    env.safe_controller.correct = capture_correct
    rows = []
    try:
        while True:
            state = env._ego_state()
            context = env._road_cbf_context()
            lcf, rcf = lane_change_feasibility(env.road, env.vehicle)
            with torch.no_grad():
                mean = agent.actor(torch.as_tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)).squeeze(0).cpu().numpy()
            policy_action = agent.select_action(observation, deterministic=True).action
            np.testing.assert_allclose(policy_action, np.tanh(mean), atol=1e-7)
            action = policy_action if fixed_heading is None else normalized_reference(state[2], fixed_heading)
            ref = action_to_reference(action)
            if fixed_heading is not None:
                np.testing.assert_allclose([ref.velocity, ref.heading], [state[2], fixed_heading], atol=1e-12)
            # Counterfactual solves operate on independent controllers, never env.mpc.
            cold = MPCController().solve(state, ref.velocity, ref.heading) if case == 'A' else None
            zero = MPCController().solve(state, state[2], 0.) if case == 'A' else None
            next_observation, reward, terminated, truncated, info = env.step(action)
            mpc, cbf = capture['mpc'], capture['cbf']
            reference = np.array([ref.velocity, ref.heading])
            np.testing.assert_allclose(info['mpc_control'], mpc.control)
            np.testing.assert_allclose(mpc.control + cbf.correction, info['safe_control'])
            zero_sequence = np.zeros_like(mpc.control_sequence)
            zero_cost = env.mpc._objective(zero_sequence.ravel(), state, reference)
            guess = capture['initial_guess']
            row = {
                'case': case, 'step': len(rows), 'time': len(rows) * SIMULATION_DT,
                'ego_x': state[0], 'ego_y': state[1], 'ego_speed': state[2], 'ego_heading': state[3],
                'lane': str(env.road.network.get_closest_lane_index(state[:2], state[3])),
                'lcf': lcf, 'rcf': rcf, 'h1': left_road_barrier(state, context), 'h2': right_road_barrier(state, context),
                'actor_mean_0': mean[0], 'actor_mean_1': mean[1],
                'policy_action_0': policy_action[0], 'policy_action_1': policy_action[1],
                'normalized_action_0': action[0], 'normalized_action_1': action[1],
                'v_ref': ref.velocity, 'theta_ref': ref.heading,
                'heading_error_ref_minus_current': ref.heading - state[3],
                'heading_error_wrapped_diagnostic_only': np.arctan2(np.sin(ref.heading-state[3]), np.cos(ref.heading-state[3])),
                **cost_components(mpc, reference), 'cost_total': mpc.objective_value,
                'zero_control_cost': zero_cost,
                'initial_guess_cost': env.mpc._objective(guess.ravel(), state, reference),
                'warm_initial_acceleration': guess[0, 0], 'warm_initial_steering': guess[0, 1],
                'mpc_acceleration': mpc.control[0], 'mpc_steering': mpc.control[1],
                'mpc_success': mpc.success, 'mpc_message': mpc.message,
                'cold_steering': cold.control[1] if cold else '',
                'cold_cost': cold.objective_value if cold else '', 'cold_success': cold.success if cold else '',
                'same_state_zero_ref_steering': zero.control[1] if zero else '',
                'cbf_delta_acceleration': cbf.correction[0], 'cbf_delta_steering': cbf.correction[1],
                'safe_acceleration': cbf.corrected_control[0], 'safe_steering': cbf.corrected_control[1],
                'nominal_constraint_min': capture['nominal_constraint_min'],
                'safe_constraint_min': cbf.minimum_constraint_value, 'cbf_success': cbf.success,
                'cbf_intervention': info['cbf_intervention'],
                'control_applied': info.get('control_applied', True),
                'applied_acceleration': cbf.corrected_control[0] if info.get('control_applied', True) else '',
                'applied_steering': cbf.corrected_control[1] if info.get('control_applied', True) else '',
                'next_x': float(env.vehicle.position[0]), 'next_y': float(env.vehicle.position[1]),
                'next_heading': float(env.vehicle.heading), 'reward': reward,
                'terminated': terminated, 'truncated': truncated,
                'success': info['success'], 'collision': info['collision'],
                'road_departure': info['road_departure'], 'termination_reason': info['termination_reason'],
                'observation': json.dumps(observation.tolist()),
            }
            rows.append(row)
            if args.verbose:
                print(f"{case} t={row['time']:5.1f} mean={mean[1]:+.3f} ref={ref.heading:+.3f} heading={state[3]:+.3f} mpc={mpc.control[1]:+.3f} cbf={cbf.correction[1]:+.3f}", flush=True)
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
        write_csv(output / f'{case}.csv', rows)
    saturation = [abs(r['mpc_steering']) >= CONTROL_UPPER_BOUND[1] - 1e-4 for r in rows]
    nominal_safe = [r for r in rows if r['nominal_constraint_min'] >= 0]
    summary = {
        'case': case, 'steps': len(rows), 'saturation_fraction': float(np.mean(saturation)),
        'mpc_failures': sum(not r['mpc_success'] for r in rows),
        'cbf_failures': sum(not r['cbf_success'] for r in rows),
        'rejected_controls': sum(not r['control_applied'] for r in rows),
        'unsafe_controls_applied': sum(r['control_applied'] and not r['cbf_success'] for r in rows),
        'interventions': sum(r['cbf_intervention'] for r in rows),
        'theta_ref_range': [min(r['theta_ref'] for r in rows), max(r['theta_ref'] for r in rows)],
        'steering_range': [min(r['mpc_steering'] for r in rows), max(r['mpc_steering'] for r in rows)],
        'nominal_safe_steps': len(nominal_safe),
        'max_correction_when_nominal_safe': max((float(np.hypot(r['cbf_delta_acceleration'], r['cbf_delta_steering'])) for r in nominal_safe), default=None),
        'final_x': rows[-1]['next_x'], 'final_y': rows[-1]['next_y'],
        'termination_reason': rows[-1]['termination_reason'],
    }
    if case == 'A':
        summary['max_warm_cold_steering_difference'] = max(abs(r['mpc_steering'] - r['cold_steering']) for r in rows)
        summary['cold_failures'] = sum(not r['cold_success'] for r in rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'results/models/bestPolicy.pt')
    parser.add_argument('--speed', type=float, choices=(5., 10.), default=5.)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnostics/mpcReference')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    agent = PPOAgent(device='cpu', seed=args.seed)
    agent.load(args.model, load_optimizer=False)
    agent.actor.eval()
    report = {
        'seed': args.seed, 'speed': args.speed, 'model': str(args.model.resolve()),
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'checkpoint_environment_version': 'not recorded; compatibility cannot be certified',
        'production_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in ('rl', 'mpc', 'cbf', 'environment', 'parameters', 'scenario', 'surrounding_vehicle_model')
                              for p in (ROOT / folder).glob('*.py')},
        'isolated_checks': isolated_checks(),
        'cases': [run_case(agent, case, heading, args, args.output)
                  for case, heading in [('A', None), ('B', 0.), ('C', .1), ('D', -.1)]],
    }
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
