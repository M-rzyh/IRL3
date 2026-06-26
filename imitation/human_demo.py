#!/usr/bin/env python3
"""
Human demonstration collection for GAIL (IRL3/imitation).

Saves demos as imitation.data.types.TrajectoryWithRew objects via
imitation.data.serialize.save(), directly loadable by GAIL training scripts.

Usage:
    # Web mode (default) — browser UI, works over SSH tunnel, no X11 needed
    python human_demo.py --env LunarLander-v2 --mode web --port 8080

    # Visual mode — renders the game via pygame, needs X11/display
    python human_demo.py --env LunarLander-v2 --mode visual

    # Curses mode — text-only, works over SSH without X11
    python human_demo.py --env LunarLander-v2 --mode curses

    # Text mode — step-by-step prompts
    python human_demo.py --env LunarLander-v2 --mode text

Controls (all modes):
    W / Up    : main engine ON  (discrete: action 2)
    A / Left  : thrust left     (discrete: action 1)
    D / Right : thrust right    (discrete: action 3)
    S / Down  : no-op / coast   (discrete: action 0)
    SPACE     : end current episode (save it)
    X         : discard current episode
    Q         : save all & quit
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Action presets for discrete LunarLander-v2
# 0=noop, 1=fire left, 2=fire main, 3=fire right
# ---------------------------------------------------------------------------
PRESETS_DISCRETE = {
    'w': 2,   # main engine
    'a': 1,   # left engine
    'd': 3,   # right engine
    's': 0,   # noop
    'none': 0,
}

# Action presets for continuous envs
PRESETS_CONTINUOUS = {
    'w':     np.array([ 1.0,  0.0], dtype=np.float32),
    'a':     np.array([ 0.0, -1.0], dtype=np.float32),
    'd':     np.array([ 0.0,  1.0], dtype=np.float32),
    's':     np.array([ 0.0,  0.0], dtype=np.float32),
    'wa':    np.array([ 1.0, -1.0], dtype=np.float32),
    'wd':    np.array([ 1.0,  1.0], dtype=np.float32),
    'none':  np.array([ 0.0,  0.0], dtype=np.float32),
}


def _state_string(obs, step, ep_reward, act=None):
    """One-line state summary for LunarLander (obs_dim=8)."""
    lines = []
    if len(obs) == 8:
        lines.append(f"  Step {step:4d}  |  Ep reward: {ep_reward:+8.2f}")
        lines.append(f"  Pos  x={obs[0]:+.3f}  y={obs[1]:+.3f}")
        lines.append(f"  Vel  vx={obs[2]:+.3f}  vy={obs[3]:+.3f}")
        lines.append(f"  Angle {obs[4]:+.3f} rad ({np.degrees(obs[4]):+.1f} deg)   angvel={obs[5]:+.3f}")
        l, r = 'Y' if obs[6] > 0.5 else 'N', 'Y' if obs[7] > 0.5 else 'N'
        lines.append(f"  Legs  L={l}  R={r}")
        if act is not None:
            if isinstance(act, (int, np.integer)):
                names = {0: 'noop', 1: 'left', 2: 'main', 3: 'right'}
                lines.append(f"  Last action: {act} ({names.get(int(act), '?')})")
            else:
                lines.append(f"  Last action: main={act[0]:+.2f}  lat={act[1]:+.2f}")
    else:
        n = min(8, len(obs))
        vals = ' '.join(f'{v:+.3f}' for v in obs[:n])
        lines.append(f"  Step {step:4d}  |  Ep reward: {ep_reward:+.2f}")
        lines.append(f"  Obs[:{n}]: {vals}{'...' if len(obs)>n else ''}")
        if act is not None:
            lines.append(f"  Action: {act}")
    return '\n'.join(lines)


def _is_discrete(env):
    """Check if action space is discrete."""
    import gymnasium
    return isinstance(env.action_space, gymnasium.spaces.Discrete)


# ---------------------------------------------------------------------------
# Web mode — browser-based UI, streams frames over HTTP
# ---------------------------------------------------------------------------

_WEB_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Human Demo Collection</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #ddd; font-family: monospace;
         display: flex; flex-direction: column; align-items: center;
         padding: 15px; min-height: 100vh; }
  h2 { margin-bottom: 8px; color: #8cf; }
  #status { font-size: 16px; margin: 8px 0; min-height: 22px; color: #ff0; }
  #frame { border: 2px solid #444; background: #000; display: block; }
  #controls { color: #888; margin: 10px 0; font-size: 13px; }
  .k { background: #282828; padding: 2px 7px; border-radius: 3px;
       border: 1px solid #555; color: #ccc; font-weight: bold; }
  #keys { color: #555; margin-top: 4px; font-size: 12px; }
  #alert { color: #f44; font-size: 14px; margin: 6px 0; min-height: 18px; }
</style>
</head>
<body>
  <h2>LunarLander Demo Collection</h2>
  <div id="alert">Click here first to capture keyboard, then press SPACE to start!</div>
  <div id="status">Connecting...</div>
  <canvas id="frame" width="600" height="400" style="width:600px; height:400px;"></canvas>
  <div id="controls">
    <span class="k">W</span>/<span class="k">&uarr;</span> up &nbsp;
    <span class="k">A</span>/<span class="k">&larr;</span> left &nbsp;
    <span class="k">D</span>/<span class="k">&rarr;</span> right &nbsp;
    <span class="k">S</span>/<span class="k">&darr;</span> coast
    &nbsp;|&nbsp;
    <span class="k">SPACE</span> end episode &nbsp;
    <span class="k">X</span> discard &nbsp;
    <span class="k">Q</span> quit &amp; save
  </div>
  <div id="keys">Keys: none</div>

<script>
const canvas = document.getElementById('frame');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const keysEl = document.getElementById('keys');
const alertEl = document.getElementById('alert');

let hasFocus = false;
window.addEventListener('focus', () => { hasFocus = true; alertEl.textContent = ''; });
window.addEventListener('blur', () => { hasFocus = false; alertEl.textContent = 'Click here to capture keyboard!'; keys.clear(); });
document.addEventListener('click', () => { window.focus(); });

// --- keyboard tracking ---
let keys = new Set();
document.addEventListener('keydown', e => {
  e.preventDefault();
  if (['Space','KeyX','KeyQ','Escape'].includes(e.code)) {
    fetch('/special', {method:'POST', body:e.code});
    return;
  }
  keys.add(e.code);
});
document.addEventListener('keyup', e => { e.preventDefault(); keys.delete(e.code); });

// Send held keys at ~30 Hz
setInterval(() => {
  fetch('/keys', {method:'POST', headers:{'Content-Type':'application/json'},
                  body:JSON.stringify([...keys])});
  keysEl.textContent = 'Keys: ' + (keys.size ? [...keys].join(', ') : 'none');
}, 33);

// --- frame polling ---
let busy = false;
setInterval(() => {
  if (busy) return;
  busy = true;
  fetch('/frame?t=' + Date.now())
    .then(r => { if (!r.ok) throw 0; return r.blob(); })
    .then(blob => createImageBitmap(blob))
    .then(bmp => {
      ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      busy = false;
    })
    .catch(() => { busy = false; });
}, 33);

// --- status polling ---
setInterval(() => {
  fetch('/status').then(r=>r.json()).then(d => {
    statusEl.textContent =
      'Ep ' + d.ep + '/' + d.max_ep +
      '  |  Step ' + d.step +
      '  |  Reward: ' + d.reward.toFixed(2) +
      '  |  Saved: ' + d.saved +
      '  |  ' + d.msg;
  }).catch(()=>{});
}, 200);
</script>
</body>
</html>
"""


def _run_web(env_id, max_episodes, fps, discrete, seed, port):
    """Run demo collection with a browser-based UI."""
    import gymnasium as gym
    import threading
    import json
    import io
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    from PIL import Image

    env = gym.make(env_id, render_mode='rgb_array')

    # Shared state between HTTP server and game loop
    lock = threading.Lock()
    shared = {
        'frame_bytes': b'',
        'pressed_keys': set(),
        'special': None,
        'ep': 0, 'max_ep': max_episodes,
        'step': 0, 'reward': 0.0,
        'saved': 0, 'msg': 'Starting...',
    }

    def encode_frame(frame):
        # 2x downsample for fast encoding — browser CSS scales it back up
        small = frame[::2, ::2]
        img = Image.fromarray(small)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=50)
        return buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == '/':
                self._respond(200, 'text/html', _WEB_HTML.encode())
            elif self.path.startswith('/frame'):
                with lock:
                    data = shared['frame_bytes']
                if data:
                    self._respond(200, 'image/jpeg', data,
                                  extra={'Cache-Control': 'no-store'})
                else:
                    self.send_response(204)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
            elif self.path == '/status':
                with lock:
                    info = {k: shared[k] for k in
                            ('ep','max_ep','step','reward','saved','msg')}
                self._respond(200, 'application/json',
                              json.dumps(info).encode())
            else:
                self.send_response(404)
                self.send_header('Content-Length', '0')
                self.end_headers()

        def do_POST(self):
            cl = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(cl) if cl else b''
            if self.path == '/keys':
                data = json.loads(body) if body else []
                with lock:
                    shared['pressed_keys'] = set(data)
            elif self.path == '/special':
                with lock:
                    shared['special'] = body.decode().strip() if body else None
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def _respond(self, code, ctype, data, extra=None):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            if extra:
                for k, v in extra.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass  # suppress request logs

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedServer(('0.0.0.0', port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    print(f"\n{'='*60}")
    print(f"  Web UI running on port {port}")
    print(f"  If on a remote cluster, open a tunnel first:")
    print(f"    ssh -L {port}:localhost:{port} <cluster>")
    print(f"  Then open: http://localhost:{port}")
    print(f"{'='*60}\n")

    # ---- game loop (runs in main thread) ----
    episodes = []
    ep_times = []
    all_attempts = []  # log ALL episodes (saved + discarded) with timing

    ep = 0
    while ep < max_episodes:
        obs, _ = env.reset(seed=seed + ep)
        frame = env.render()
        with lock:
            shared['frame_bytes'] = encode_frame(frame)
            shared['ep'] = ep + 1
            shared['step'] = 0
            shared['reward'] = 0.0
            shared['msg'] = 'Press SPACE to start episode'
            shared['special'] = None

        # --- Wait for SPACE before starting the episode ---
        print(f"\n  Episode {ep+1}/{max_episodes}  |  Waiting for SPACE to start...")
        waiting = True
        while waiting:
            with lock:
                special = shared['special']
                shared['special'] = None
            if special == 'Space':
                waiting = False
            elif special in ('KeyQ', 'Escape'):
                env.close()
                server.shutdown()
                return episodes, ep_times, all_attempts
            time.sleep(0.05)

        ep_start = time.time()
        with lock:
            shared['msg'] = 'Playing — use W/A/D/S'

        ep_obs, ep_act, ep_rew = [obs.copy()], [], []
        step = 0
        ep_reward = 0.0
        done = False
        discard = False
        terminal = False

        while not done:
            with lock:
                special = shared['special']
                shared['special'] = None
                keys = set(shared['pressed_keys'])

            # Handle special keys
            if special in ('KeyQ', 'Escape'):
                if len(ep_act) > 0 and not discard:
                    episodes.append((ep_obs, ep_act, ep_rew, True))
                    ep_times.append(time.time() - ep_start)
                with lock:
                    shared['msg'] = 'Quit — saving...'
                env.close()
                server.shutdown()
                return episodes, ep_times, all_attempts
            elif special == 'KeyX':
                discard = True
                break
            elif special == 'Space':
                done = True
                break

            # Map held keys to action (no key = action 0 = coast/do nothing)
            if discrete:
                if 'KeyW' in keys or 'ArrowUp' in keys:
                    action = 2
                elif 'KeyA' in keys or 'ArrowLeft' in keys:
                    action = 1
                elif 'KeyD' in keys or 'ArrowRight' in keys:
                    action = 3
                else:
                    action = 0
            else:
                act_dim = env.action_space.shape[0]
                action = np.zeros(act_dim, dtype=np.float32)
                if 'KeyW' in keys or 'ArrowUp' in keys:
                    action[0] = 1.0
                if 'KeyA' in keys or 'ArrowLeft' in keys:
                    if act_dim > 1:
                        action[1] = -1.0
                if 'KeyD' in keys or 'ArrowRight' in keys:
                    if act_dim > 1:
                        action[1] = 1.0

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_act.append(action if discrete else action.copy())
            ep_rew.append(reward)
            obs = next_obs
            ep_obs.append(obs.copy())
            step += 1
            ep_reward += reward

            # Render + encode every 2nd step to save CPU
            if step % 2 == 0:
                frame = env.render()
                with lock:
                    shared['frame_bytes'] = encode_frame(frame)
                    shared['step'] = step
                    shared['reward'] = ep_reward
            else:
                with lock:
                    shared['step'] = step
                    shared['reward'] = ep_reward

            if terminated or truncated:
                terminal = terminated
                done = True

            time.sleep(1.0 / fps)

        ep_end = time.time()

        if discard:
            all_attempts.append({
                'status': 'discarded_during',
                'duration_sec': ep_end - ep_start,
                'reward': ep_reward,
                'steps': step,
            })
            with lock:
                shared['msg'] = f'Episode {ep+1} discarded'
            print(f"  Episode {ep+1} discarded.")
            time.sleep(1.5)
            continue

        if len(ep_act) > 0:
            # Ask user whether to save or discard this episode
            with lock:
                shared['msg'] = (f'Episode done: {step} steps, reward={ep_reward:+.2f}'
                                 f'  |  SPACE=save  X=discard')
            print(f"  Episode {ep+1}: {step} steps, reward={ep_reward:+.2f}  — save or discard?")

            deciding = True
            save_it = False
            while deciding:
                with lock:
                    special = shared['special']
                    shared['special'] = None
                if special == 'Space':
                    save_it = True
                    deciding = False
                elif special == 'KeyX':
                    deciding = False
                elif special in ('KeyQ', 'Escape'):
                    # Quit without saving this episode
                    env.close()
                    server.shutdown()
                    return episodes, ep_times, all_attempts
                time.sleep(0.05)

            if save_it:
                episodes.append((ep_obs, ep_act, ep_rew, terminal))
                ep_times.append(ep_end - ep_start)
                all_attempts.append({
                    'status': 'saved',
                    'duration_sec': ep_end - ep_start,
                    'reward': ep_reward,
                    'steps': step,
                })
                with lock:
                    shared['saved'] = len(episodes)
                    shared['msg'] = f'Saved! ({len(episodes)}/{max_episodes})'
                print(f"  -> Saved ({len(episodes)}/{max_episodes})")
            else:
                all_attempts.append({
                    'status': 'discarded_after',
                    'duration_sec': ep_end - ep_start,
                    'reward': ep_reward,
                    'steps': step,
                })
                with lock:
                    shared['msg'] = 'Discarded — try again'
                print(f"  -> Discarded")
                time.sleep(1.0)
                continue  # don't increment ep, retry

        time.sleep(1.5)
        ep += 1

    with lock:
        shared['msg'] = 'Done — saving...'
    env.close()
    server.shutdown()
    return episodes, ep_times, all_attempts


# ---------------------------------------------------------------------------
# Visual mode — pygame rendering + keyboard input (see the actual game)
# ---------------------------------------------------------------------------

def _blit_frame(screen, frame):
    """Blit an rgb_array frame (H,W,3 uint8) onto a pygame surface."""
    import pygame
    # frame is (H, W, 3) — pygame wants (W, H, 3) for surfarray
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surf, (0, 0))


def _draw_overlay(screen, text_lines):
    """Draw text overlay on the pygame surface."""
    import pygame
    font = pygame.font.SysFont('monospace', 22, bold=True)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))
    y = screen.get_height() // 2 - len(text_lines) * 15
    for line in text_lines:
        text_surf = font.render(line, True, (255, 255, 100))
        x = (screen.get_width() - text_surf.get_width()) // 2
        screen.blit(text_surf, (x, y))
        y += 32
    pygame.display.flip()


def _wait_for_start(screen, frame, ep_num, max_ep, saved):
    """Show 'Press SPACE to start' overlay and block until SPACE is pressed."""
    import pygame
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return 'quit'
                elif event.key == pygame.K_SPACE:
                    return 'start'
        _blit_frame(screen, frame)
        _draw_overlay(screen, [
            f"Episode {ep_num}/{max_ep}  |  Saved: {saved}",
            "",
            "Press SPACE to start",
            "",
            "W=up  A=left  D=right  S=coast",
            "SPACE=end episode  X=discard  Q=quit",
        ])
        clock.tick(15)


def _run_visual(env_id, max_episodes, fps, discrete, seed):
    """Render the game visually and capture keyboard input via pygame.

    Uses render_mode='rgb_array' and blits frames to a pygame window manually.
    This avoids OpenGL/GLX issues on headless displays (VNC, Xvfb).

    The environment only steps when you are actively pressing a control key
    (W/A/D/S or arrows). If no key is held, the game pauses — the lander
    stays frozen until you act. Each episode starts only after you press SPACE.
    """
    import pygame
    import gymnasium as gym

    # Use rgb_array to avoid GLX/OpenGL issues, then blit to our own pygame window
    env = gym.make(env_id, render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    frame = env.render()  # (H, W, 3) uint8

    # Create pygame window matching frame size
    pygame.init()
    h, w = frame.shape[:2]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f'Human Demo - {env_id}')

    clock = pygame.time.Clock()
    episodes = []
    ep_times = []

    ep = 0
    while ep < max_episodes:
        obs, _ = env.reset()
        frame = env.render()

        # --- Wait for user to press SPACE before starting ---
        print(f"\n  Episode {ep+1}/{max_episodes}  |  Saved: {len(episodes)}")
        print("  Press SPACE in the game window to start...")
        result = _wait_for_start(screen, frame, ep + 1, max_episodes, len(episodes))
        if result == 'quit':
            env.close()
            pygame.quit()
            return episodes, ep_times

        ep_start = time.time()
        ep_obs, ep_act, ep_rew = [obs.copy()], [], []
        step = 0
        ep_reward = 0.0
        done = False
        discard = False
        terminal = False

        print("  GO! Controls: W=up  A=left  D=right  S=coast  SPACE=end  X=discard  Q=quit")

        while not done:
            quit_all = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_all = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        quit_all = True
                    elif event.key == pygame.K_x:
                        discard = True
                    elif event.key == pygame.K_SPACE:
                        done = True

            if quit_all:
                if len(ep_act) > 0 and not discard:
                    episodes.append((ep_obs, ep_act, ep_rew, True))
                    ep_times.append(time.time() - ep_start)
                env.close()
                pygame.quit()
                return episodes, ep_times

            if discard or done:
                break

            # Read held keys — only step the env if a control key is pressed.
            # If no key is held, the game pauses (lander freezes in place).
            keys = pygame.key.get_pressed()
            has_input = (
                keys[pygame.K_w] or keys[pygame.K_UP] or
                keys[pygame.K_a] or keys[pygame.K_LEFT] or
                keys[pygame.K_d] or keys[pygame.K_RIGHT] or
                keys[pygame.K_s] or keys[pygame.K_DOWN]
            )

            if not has_input:
                # No key pressed — just show the last frame, don't step
                _blit_frame(screen, frame)
                pygame.display.flip()
                clock.tick(fps)
                continue

            # Map held keys to action
            if discrete:
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    action = 2  # main engine
                elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    action = 1  # left
                elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    action = 3  # right
                else:
                    action = 0  # noop (S key)
            else:
                act_dim = env.action_space.shape[0]
                action = np.zeros(act_dim, dtype=np.float32)
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    action[0] = 1.0
                if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    if act_dim > 1:
                        action[1] = -1.0
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    if act_dim > 1:
                        action[1] = 1.0

            next_obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            _blit_frame(screen, frame)
            pygame.display.flip()

            ep_act.append(action if discrete else action.copy())
            ep_rew.append(reward)

            obs = next_obs
            ep_obs.append(obs.copy())
            step += 1
            ep_reward += reward

            if terminated or truncated:
                terminal = terminated
                done = True

            clock.tick(fps)

        ep_end = time.time()

        if discard:
            print(f"  Episode {ep+1} discarded.")
            continue

        if len(ep_act) > 0:
            episodes.append((ep_obs, ep_act, ep_rew, terminal))
            ep_times.append(ep_end - ep_start)
            print(f"  Episode {ep+1} done: {step} steps, reward={ep_reward:+.2f}")

        ep += 1

    env.close()
    pygame.quit()
    return episodes, ep_times


# ---------------------------------------------------------------------------
# Curses-based real-time demo collection
# ---------------------------------------------------------------------------

def _run_curses(env, max_episodes, fps, discrete):
    import curses

    episodes = []   # list of (obs_list, act_list, rew_list, done_flag)
    ep_times = []

    def _inner(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(1000 / fps))

        ep = 0
        while ep < max_episodes:
            ep_start = time.time()
            obs, _ = env.reset()
            ep_obs, ep_act, ep_rew = [obs.copy()], [], []
            step = 0
            ep_reward = 0.0
            done = False
            last_act = 0 if discrete else np.zeros(env.action_space.shape[0], dtype=np.float32)
            discard = False
            terminal = False

            while not done:
                stdscr.clear()
                stdscr.addstr(0, 0, f"{'=' * 56}")
                mode_str = "DISCRETE" if discrete else "CONTINUOUS"
                stdscr.addstr(1, 0, f"  DEMO [{mode_str}]  |  Ep {ep+1}/{max_episodes}  |  Saved: {len(episodes)}")
                stdscr.addstr(2, 0, f"{'=' * 56}")
                for i, line in enumerate(_state_string(obs, step, ep_reward, last_act).split('\n')):
                    stdscr.addstr(4 + i, 0, line)
                row = 4 + i + 2
                stdscr.addstr(row, 0, "  W=up  A=left  D=right  S=coast")
                stdscr.addstr(row+1, 0, "  SPACE=end episode  X=discard  Q=quit")
                stdscr.refresh()

                key = stdscr.getch()

                quit_all = False

                if key == ord('q') or key == ord('Q'):
                    quit_all = True
                    break
                elif key == ord('x') or key == ord('X'):
                    discard = True
                    break
                elif key == ord(' '):
                    done = True
                    break

                # Map key to action
                if discrete:
                    if key == ord('w') or key == curses.KEY_UP:
                        action = 2
                    elif key == ord('a') or key == curses.KEY_LEFT:
                        action = 1
                    elif key == ord('d') or key == curses.KEY_RIGHT:
                        action = 3
                    else:
                        action = 0
                else:
                    act_dim = env.action_space.shape[0]
                    action = np.zeros(act_dim, dtype=np.float32)
                    if key == ord('w') or key == curses.KEY_UP:
                        action[0] = 1.0
                    elif key == ord('a') or key == curses.KEY_LEFT:
                        if act_dim > 1:
                            action[1] = -1.0
                    elif key == ord('d') or key == curses.KEY_RIGHT:
                        if act_dim > 1:
                            action[1] = 1.0

                next_obs, reward, terminated, truncated, info = env.step(action)
                last_act = action if discrete else action.copy()
                ep_act.append(action if discrete else action.copy())
                ep_rew.append(reward)

                obs = next_obs
                ep_obs.append(obs.copy())
                step += 1
                ep_reward += reward

                if terminated or truncated:
                    terminal = terminated
                    done = True

            ep_end = time.time()

            if quit_all:
                if len(ep_act) > 0 and not discard:
                    episodes.append((ep_obs, ep_act, ep_rew, True))
                    ep_times.append(ep_end - ep_start)
                break

            if discard:
                stdscr.clear()
                stdscr.addstr(0, 0, f"  Episode {ep+1} discarded.")
                stdscr.refresh()
                curses.napms(600)
                continue

            if len(ep_act) > 0:
                episodes.append((ep_obs, ep_act, ep_rew, terminal))
                ep_times.append(ep_end - ep_start)

            stdscr.clear()
            stdscr.addstr(0, 0, f"  Episode {ep+1} done: {step} steps, reward={ep_reward:+.2f}")
            stdscr.addstr(1, 0, f"  Total saved: {len(episodes)} episodes")
            stdscr.refresh()
            curses.napms(1200)
            ep += 1

    curses.wrapper(_inner)
    return episodes, ep_times


# ---------------------------------------------------------------------------
# Text-based fallback
# ---------------------------------------------------------------------------

def _run_text(env, max_episodes, discrete):
    episodes = []
    ep_times = []

    for ep in range(max_episodes):
        ep_start = time.time()
        obs, _ = env.reset()
        ep_obs, ep_act, ep_rew = [obs.copy()], [], []
        step = 0
        ep_reward = 0.0
        done = False
        discard = False
        terminal = False

        print(f"\n{'=' * 56}")
        print(f"  Episode {ep+1}/{max_episodes}  |  Saved so far: {len(episodes)}")
        act_type = "DISCRETE (0=noop, 1=left, 2=main, 3=right)" if discrete else "CONTINUOUS"
        print(f"  Action space: {act_type}")
        print(f"{'=' * 56}")

        while not done:
            print(_state_string(obs, step, ep_reward))
            if discrete:
                print("  Enter: w/a/d/s or action index (0-3), space=end, x=discard, q=quit")
            else:
                print("  Enter: w/a/d/s, or 'main,lat' (e.g. 0.5,-1), space=end, x=discard, q=quit")

            try:
                raw = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = 'q'

            if raw == 'q':
                if len(ep_act) > 0 and not discard:
                    episodes.append((ep_obs, ep_act, ep_rew, True))
                    ep_times.append(time.time() - ep_start)
                return episodes, ep_times
            elif raw == 'x':
                discard = True
                print("  Discarded.")
                break
            elif raw in ('', 'space', ' '):
                break

            if discrete:
                if raw in PRESETS_DISCRETE:
                    action = PRESETS_DISCRETE[raw]
                else:
                    try:
                        action = int(raw)
                        if action < 0 or action >= env.action_space.n:
                            print(f"  (out of range, using noop)")
                            action = 0
                    except ValueError:
                        print("  (unrecognized, using noop)")
                        action = 0
            else:
                act_dim = env.action_space.shape[0]
                if raw in PRESETS_CONTINUOUS:
                    action = PRESETS_CONTINUOUS[raw].copy()
                    if act_dim != 2:
                        action = np.zeros(act_dim, dtype=np.float32)
                        action[0] = PRESETS_CONTINUOUS[raw][0]
                else:
                    try:
                        vals = [float(x) for x in raw.replace(' ', ',').split(',')]
                        action = np.zeros(act_dim, dtype=np.float32)
                        for j, v in enumerate(vals[:act_dim]):
                            action[j] = np.clip(v, -1.0, 1.0)
                    except ValueError:
                        print("  (unrecognized, using coast)")
                        action = np.zeros(act_dim, dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_act.append(action if discrete else action.copy())
            ep_rew.append(reward)
            obs = next_obs
            ep_obs.append(obs.copy())
            step += 1
            ep_reward += reward

            if terminated or truncated:
                terminal = terminated
                done = True

        ep_end = time.time()

        if discard:
            continue

        if len(ep_act) > 0:
            episodes.append((ep_obs, ep_act, ep_rew, terminal))
            ep_times.append(ep_end - ep_start)
            print(f"  Saved episode {ep+1}: {step} steps, reward={ep_reward:+.2f}")

    return episodes, ep_times


# ---------------------------------------------------------------------------
# Save as imitation TrajectoryWithRew via serialize.save()
# ---------------------------------------------------------------------------

def save_demos(episodes, ep_times, output_dir, all_attempts=None):
    """Convert collected episodes to TrajectoryWithRew and save via imitation.data.serialize."""
    from imitation.data.types import TrajectoryWithRew
    from imitation.data import serialize

    trajectories = []
    for ep_obs, ep_act, ep_rew, terminal in episodes:
        obs = np.array(ep_obs, dtype=np.float32)        # (T+1, obs_dim)
        if isinstance(ep_act[0], (int, np.integer)):
            acts = np.array(ep_act, dtype=np.int64)      # (T,)
        else:
            acts = np.array(ep_act, dtype=np.float32)     # (T, act_dim)
        rews = np.array(ep_rew, dtype=np.float64)        # (T,)

        traj = TrajectoryWithRew(
            obs=obs,
            acts=acts,
            rews=rews,
            infos=None,
            terminal=terminal,
        )
        trajectories.append(traj)

    os.makedirs(output_dir, exist_ok=True)
    serialize.save(output_dir, trajectories)

    total_steps = sum(len(t) for t in trajectories)
    total_reward = sum(float(t.rews.sum()) for t in trajectories)
    print(f"\nSaved {len(trajectories)} episodes ({total_steps} steps, "
          f"total reward={total_reward:+.1f}) -> {output_dir}")

    # Timing CSV for saved episodes only
    csv_path = os.path.join(output_dir, 'timing.csv')
    cumulative = 0.0
    with open(csv_path, 'w') as f:
        f.write("episode_idx,duration_sec,cumulative_sec,ep_reward,ep_length\n")
        for i, (dt, (_, _, ep_rew, _)) in enumerate(zip(ep_times, episodes)):
            cumulative += dt
            reward = sum(ep_rew)
            length = len(ep_rew)
            f.write(f"{i},{dt:.3f},{cumulative:.3f},{reward:.2f},{length}\n")
    print(f"Saved timing CSV -> {csv_path}")
    print(f"Total demo collection time (saved only): {cumulative:.1f} sec")

    # Full session log: ALL episodes (saved + discarded) with timing
    if all_attempts:
        log_path = os.path.join(output_dir, 'all_episodes_log.csv')
        total_cumulative = 0.0
        with open(log_path, 'w') as f:
            f.write("attempt,status,duration_sec,cumulative_sec,reward,steps\n")
            for i, att in enumerate(all_attempts):
                total_cumulative += att['duration_sec']
                f.write(f"{i},{att['status']},{att['duration_sec']:.3f},"
                        f"{total_cumulative:.3f},{att['reward']:.2f},{att['steps']}\n")
        n_saved = sum(1 for a in all_attempts if a['status'] == 'saved')
        n_discarded = sum(1 for a in all_attempts if a['status'] != 'saved')
        print(f"Saved full session log -> {log_path}")
        print(f"  Total attempts: {len(all_attempts)} ({n_saved} saved, {n_discarded} discarded)")
        print(f"  Total wall-clock time (all attempts): {total_cumulative:.1f} sec")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect human demonstrations for GAIL (imitation library format)")
    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='Gymnasium environment ID (default: LunarLander-v2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: demos/<env>/human_demos)')
    parser.add_argument('--max_episodes', type=int, default=20,
                        help='Max episodes to collect')
    parser.add_argument('--fps', type=int, default=10,
                        help='Target FPS for curses mode (default: 10)')
    parser.add_argument('--mode', type=str, choices=['web', 'visual', 'curses', 'text'], default='web',
                        help='Input mode: web (browser UI, default), visual (pygame window), curses (text), text (step-by-step)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for web mode (default: 8080)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import gymnasium as gym

    # Probe env for info display
    probe_env = gym.make(args.env)
    discrete = _is_discrete(probe_env)
    if discrete:
        print(f"Environment: {args.env}  (obs={probe_env.observation_space.shape[0]}, "
              f"discrete actions: {probe_env.action_space.n})")
    else:
        print(f"Environment: {args.env}  (obs={probe_env.observation_space.shape[0]}, "
              f"act={probe_env.action_space.shape[0]})")
    print(f"Mode: {args.mode}   Max episodes: {args.max_episodes}")
    probe_env.close()

    all_attempts = []

    if args.mode == 'web':
        episodes, ep_times, all_attempts = _run_web(
            args.env, args.max_episodes, args.fps, discrete, args.seed, args.port)
    elif args.mode == 'visual':
        episodes, ep_times = _run_visual(
            args.env, args.max_episodes, args.fps, discrete, args.seed)
    elif args.mode == 'curses':
        env = gym.make(args.env)
        env.reset(seed=args.seed)
        episodes, ep_times = _run_curses(env, args.max_episodes, args.fps, discrete)
        env.close()
    else:
        env = gym.make(args.env)
        env.reset(seed=args.seed)
        episodes, ep_times = _run_text(env, args.max_episodes, discrete)
        env.close()

    if len(episodes) == 0:
        print("No demos collected.")
        return

    # Auto-number: find next session_N folder that doesn't exist
    base_dir = args.output or f"demos/{args.env}/human_demos"
    session = 1
    while os.path.exists(os.path.join(base_dir, f"session_{session}")):
        session += 1
    output_dir = os.path.join(base_dir, f"session_{session}")

    save_demos(episodes, ep_times, output_dir, all_attempts=all_attempts)
    print(f"\n  Next run will save to: {os.path.join(base_dir, f'session_{session+1}')}")


if __name__ == '__main__':
    main()
